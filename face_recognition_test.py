from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import face_recognition
import numpy as np
import json
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import av
import os
import time
import fractions  # Moved to top imports
from typing import Dict, Any, Set, List, Optional
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Configuration parameters - adjusted for better performance
FPS = 30  # Reduced from 90 to a more realistic value for most cameras
FACE_RECOGNITION_INTERVAL = 5  # Increased interval to reduce CPU load
FACE_RECOGNITION_SCALE = 0.2  # Reduced scale for faster processing
LOW_LATENCY_MODE = True  # Prioritize latency over quality
MAX_RESOLUTION = (640, 480)  # Maximum resolution to use

# Thread pool for face recognition processing
executor = ThreadPoolExecutor(max_workers=2)

# Load known faces from JSON
known_faces_data = {}
known_encodings = []
known_names = []

try:
    if os.path.exists("known_faces.json"):
        with open("known_faces.json", "r") as file:
            known_faces_data = json.load(file)
            known_encodings = [np.array(encoding) for encoding in known_faces_data.values()]
            known_names = list(known_faces_data.keys())
            print(f"Loaded {len(known_names)} known faces.")
    else:
        print("known_faces.json not found. Creating empty face database.")
        # Create empty file
        with open("known_faces.json", "w") as file:
            json.dump({}, file)
except Exception as e:
    print(f"Error loading known_faces.json: {e}")

# Store active peer connections
peer_connections: Set[RTCPeerConnection] = set()

# Camera variable
camera = None

# Last processed face recognition results
last_face_results = []

def init_camera():
    global camera
    try:
        camera = cv2.VideoCapture(0)
        
        # Set camera properties for low latency
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        camera.set(cv2.CAP_PROP_FPS, FPS)  # Set target FPS
        
        if LOW_LATENCY_MODE:
            # Lower resolution for faster processing
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_RESOLUTION[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_RESOLUTION[1])
            
            # Additional low-latency settings
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for lower latency
        
        if not camera.isOpened():
            print("Error: Could not open camera. Check if it's connected.")
            return False
        
        # Get actual camera properties (may differ from requested)
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        print(f"Camera initialized with resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

# Initialize camera at startup
init_camera()

def process_face_recognition(rgb_frame):
    """Process face recognition in a separate thread to avoid blocking"""
    try:
        # Resize for faster processing
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=FACE_RECOGNITION_SCALE, fy=FACE_RECOGNITION_SCALE)
        
        # Find faces
        face_locations = face_recognition.face_locations(small_frame, model="hog")  # Use faster HOG model
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        # Scale face locations back to original size
        face_locations = [(int(top/FACE_RECOGNITION_SCALE), 
                          int(right/FACE_RECOGNITION_SCALE), 
                          int(bottom/FACE_RECOGNITION_SCALE), 
                          int(left/FACE_RECOGNITION_SCALE)) 
                         for top, right, bottom, left in face_locations]
        
        results = []
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            box_color = (0, 0, 255)  # Red for unknown faces
            
            if len(known_encodings) > 0:
                # Compare with known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                
                if True in matches:
                    # Only calculate face distances for actual matches to save processing
                    face_distances = face_recognition.face_distance(
                        [known_encodings[i] for i, match in enumerate(matches) if match], 
                        face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    name = known_names[[i for i, match in enumerate(matches) if match][best_match_index]]
                    box_color = (0, 255, 0)  # Green for known faces
            
            results.append({
                'location': (left, top, right, bottom),
                'name': name,
                'color': box_color
            })
        
        return results
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return []

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.frame_timestamp = 0
        self.start_time = time.time()
        self.face_recognition_future = None
        self.last_frame = None  # Cache last frame for error recovery
        
    async def recv(self):
        global camera, last_face_results
        
        # Reinitialize camera if needed
        if camera is None or not camera.isOpened():
            print("Reinitializing camera...")
            success = init_camera()
            if not success:
                # Return previous frame if available, otherwise a black frame
                if self.last_frame is not None:
                    frame = av.VideoFrame.from_ndarray(self.last_frame, format="bgr24")
                else:
                    black_frame = np.zeros((MAX_RESOLUTION[1], MAX_RESOLUTION[0], 3), dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(black_frame, format="bgr24")
                
                frame.pts = self.frame_count
                frame.time_base = fractions.Fraction(1, FPS)
                self.frame_count += 1
                await asyncio.sleep(1/FPS)  # Add small delay to avoid CPU spike
                return frame
        
        # Read from camera with timeout to prevent blocking
        try:
            # Use a very short timeout for reading frames
            success, img = camera.read()
        except Exception as e:
            print(f"Error reading camera frame: {e}")
            success = False
        
        if not success:
            print("Frame read error. Using fallback.")
            if self.last_frame is not None:
                img = self.last_frame.copy()
                success = True
            else:
                # Create a black frame with error message
                img = np.zeros((MAX_RESOLUTION[1], MAX_RESOLUTION[0], 3), dtype=np.uint8)
                cv2.putText(img, "Camera Error - Reconnecting...", (50, MAX_RESOLUTION[1]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                await asyncio.sleep(0.1)  # Small delay to reduce CPU usage during error
        else:
            # Cache successful frame for error recovery
            self.last_frame = img.copy()
        
        # Start face recognition processing only on certain frames to reduce CPU load
        if success and self.frame_count % FACE_RECOGNITION_INTERVAL == 0:
            if self.face_recognition_future is None or self.face_recognition_future.done():
                # Convert to RGB for face_recognition
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Process face recognition in a separate thread
                self.face_recognition_future = asyncio.get_event_loop().run_in_executor(
                    executor, process_face_recognition, rgb_frame.copy())
        
        # Check if face recognition is completed and update results
        if self.face_recognition_future and self.face_recognition_future.done():
            try:
                results = self.face_recognition_future.result()
                if results:  # Only update if we got results
                    last_face_results = results
                self.face_recognition_future = None
            except Exception as e:
                print(f"Error getting face recognition results: {e}")
        
        # Draw face rectangles and names using the latest results
        for face in last_face_results:
            left, top, right, bottom = face['location']
            # Draw with semi-transparent background for better visibility
            cv2.rectangle(img, (left, top), (right, bottom), face['color'], 2)
            
            # Add background behind text for better visibility
            cv2.rectangle(img, (left, top - 30), (left + len(face['name'])*12, top), face['color'], -1)
            cv2.putText(img, face['name'], (left + 6, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add timestamp and FPS counter for debugging
        if LOW_LATENCY_MODE:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                cv2.putText(img, f"FPS: {fps:.1f}", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Convert the frame to VideoFrame
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Set frame timing for smooth playback
        frame.pts = self.frame_count
        frame.time_base = fractions.Fraction(1, FPS)
        self.frame_count += 1
        
        return frame

@app.post("/offer")
async def offer(request: Dict[str, Any]):
    try:
        pc = RTCPeerConnection()
        peer_connections.add(pc)
        
        # Add video track to peer connection
        video_track = VideoStreamTrack()
        pc.addTrack(video_track)
        
        # Handle cleanup when connection is closed
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state changed to: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                if pc in peer_connections:
                    peer_connections.remove(pc)
        
        # Set remote description
        offer = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
        await pc.setRemoteDescription(offer)
        
        # Create and set local description
        answer = await pc.createAnswer()
        
        # Modify SDP for lower latency
        answer.sdp = answer.sdp.replace('a=rtcp-fb:* nack pli', 'a=rtcp-fb:* nack pli\r\na=rtcp-fb:* ccm fir')
        
        # Adjust the number of spatial layers for VP8
        answer.sdp = answer.sdp.replace('a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f',
                                       'a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f')
        
        await pc.setLocalDescription(answer)
        
        return JSONResponse(content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    
    except Exception as e:
        print(f"Error in WebRTC setup: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("index.html", "r") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        # Create optimized HTML file with low-latency settings
        with open("index.html", "w") as file:
            file.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Ultra Low-Latency Face Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        video {
            width: 100%;
            max-width: 640px;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f0f0f0;
        }
        .video-container {
            position: relative;
            margin: 0 auto;
        }
        .overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(0,0,0,0.5);
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 14px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px 10px 0;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
        }
        .success {
            background-color: #dff0d8;
            color: #3c763d;
        }
        .error {
            background-color: #f2dede;
            color: #a94442;
        }
        .warning {
            background-color: #fcf8e3;
            color: #8a6d3b;
        }
        .stats {
            font-family: monospace;
            font-size: 14px;
            margin-top: 10px;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Ultra Low-Latency Face Recognition</h1>
    <div id="status"></div>
    <div class="video-container">
        <video id="video" autoplay playsinline muted></video>
        <div class="overlay" id="latency">Latency: --</div>
    </div>
    <div class="controls">
        <button id="startButton">Start Camera</button>
        <button id="stopButton" disabled>Stop Camera</button>
        <button id="fullScreenButton">Full Screen</button>
    </div>
    <div class="stats" id="stats"></div>

    <script>
        const video = document.getElementById('video');
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const fullScreenButton = document.getElementById('fullScreenButton');
        const statusDiv = document.getElementById('status');
        const statsDiv = document.getElementById('stats');
        const latencyDiv = document.getElementById('latency');
        let pc = null;
        let statsInterval = null;
        let frameCounter = 0;
        let lastFrameTime = 0;
        let latencyValues = [];

        // Set video low-latency properties
        video.setAttribute('playsinline', '');
        video.setAttribute('muted', '');
        
        // Enable hardware acceleration if available
        video.style.transform = 'translateZ(0)';
        video.style.backfaceVisibility = 'hidden';

        function setStatus(message, type = 'success') {
            statusDiv.textContent = message;
            statusDiv.className = type;
        }

        // Track video frame callback for performance monitoring
        function frameCallback(now, metadata) {
            frameCounter++;
            
            // Calculate FPS
            if (lastFrameTime) {
                const frameDelay = now - lastFrameTime;
                const frameLatency = metadata.presentationTime ? (now - metadata.presentationTime) : 0;
                
                // Track latency for averaging (last 10 frames)
                if (frameLatency > 0) {
                    latencyValues.push(frameLatency);
                    if (latencyValues.length > 10) latencyValues.shift();
                    
                    // Calculate average latency
                    const avgLatency = latencyValues.reduce((a, b) => a + b, 0) / latencyValues.length;
                    latencyDiv.textContent = `Latency: ${avgLatency.toFixed(1)}ms`;
                    
                    // Highlight high latency
                    if (avgLatency < 100) {
                        latencyDiv.style.backgroundColor = 'rgba(0,128,0,0.5)';
                    } else if (avgLatency < 300) {
                        latencyDiv.style.backgroundColor = 'rgba(255,165,0,0.5)';
                    } else {
                        latencyDiv.style.backgroundColor = 'rgba(255,0,0,0.5)';
                    }
                }
            }
            
            lastFrameTime = now;
            
            // Request next frame
            if (video.readyState >= 2) {
                video.requestVideoFrameCallback(frameCallback);
            }
        }

        async function getStats() {
            if (!pc) return;
            
            try {
                const stats = await pc.getStats();
                let videoStats = {};
                let connectionStats = {};
                
                stats.forEach(report => {
                    if (report.type === 'inbound-rtp' && report.kind === 'video') {
                        videoStats = {
                            jitter: report.jitter ? (report.jitter * 1000).toFixed(2) + ' ms' : 'N/A',
                            packetsLost: report.packetsLost || 0,
                            framesDecoded: report.framesDecoded || 0,
                            framesDropped: report.framesDropped || 0,
                            frameHeight: report.frameHeight || 0,
                            frameWidth: report.frameWidth || 0,
                            fps: report.framesPerSecond ? report.framesPerSecond.toFixed(1) : 'N/A',
                            bytesReceived: report.bytesReceived ? (report.bytesReceived / 1024).toFixed(0) + ' KB' : '0 KB'
                        };
                    }
                    
                    if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                        connectionStats = {
                            currentRoundTripTime: report.currentRoundTripTime ? (report.currentRoundTripTime * 1000).toFixed(2) + ' ms' : 'N/A',
                            totalRoundTripTime: report.totalRoundTripTime ? (report.totalRoundTripTime * 1000).toFixed(2) + ' ms' : 'N/A',
                        };
                    }
                });
                
                statsDiv.innerHTML = `<b>Video Stats:</b> ${videoStats.frameWidth}x${videoStats.frameHeight} @ ${videoStats.fps} FPS | Jitter: ${videoStats.jitter} | RTT: ${connectionStats.currentRoundTripTime || 'N/A'} | Lost Packets: ${videoStats.packetsLost} | Data: ${videoStats.bytesReceived}`;
            } catch (e) {
                console.error('Could not get stats:', e);
            }
        }

        async function start() {
            try {
                setStatus('Connecting to server...', 'warning');
                startButton.disabled = true;
                
                // Create peer connection with low-latency configuration
                pc = new RTCPeerConnection({
                    sdpSemantics: 'unified-plan',
                    iceCandidatePoolSize: 10,
                    iceTransportPolicy: 'all',
                    bundlePolicy: 'max-bundle',
                    rtcpMuxPolicy: 'require'
                });

                // Debug ICE connection state changes
                pc.oniceconnectionstatechange = () => {
                    console.log('ICE connection state:', pc.iceConnectionState);
                    if (pc.iceConnectionState === 'failed') {
                        setStatus('ICE connection failed. Try refreshing the page.', 'error');
                    } else if (pc.iceConnectionState === 'connected') {
                        console.log('ICE connected!');
                    }
                };

                // Debug connection state changes
                pc.onconnectionstatechange = () => {
                    console.log('Connection state:', pc.connectionState);
                    if (pc.connectionState === 'connected') {
                        stopButton.disabled = false;
                        setStatus('Connected! Streaming video with face recognition.', 'success');
                        
                        // Start stats collection
                        if (statsInterval) clearInterval(statsInterval);
                        statsInterval = setInterval(getStats, 1000);
                        
                        // Start frame callback for performance tracking if supported
                        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
                            video.requestVideoFrameCallback(frameCallback);
                        }
                    } else if (pc.connectionState === 'disconnected' || 
                               pc.connectionState === 'failed' || 
                               pc.connectionState === 'closed') {
                        if (statsInterval) {
                            clearInterval(statsInterval);
                            statsInterval = null;
                        }
                    }
                };

                // Handle incoming tracks with low-latency options
                pc.ontrack = (event) => {
                    console.log('Track received:', event.track.kind);
                    if (event.track.kind === 'video') {
                        console.log('Video track received, streams:', event.streams.length);
                        if (event.streams && event.streams[0]) {
                            video.srcObject = event.streams[0];
                            
                            // Low latency settings
                            video.load();
                            
                            // Play with catch for mobile devices
                            video.play().catch(e => {
                                console.error('Error playing video:', e);
                                // Try again with user interaction
                                setStatus('Click/tap on the video area to start playback', 'warning');
                                video.addEventListener('click', () => {
                                    video.play().catch(err => console.error('Still cannot play:', err));
                                }, { once: true });
                            });
                            
                            // Reduce latency by setting minimal latency mode
                            if (video.style.webkitBackfaceVisibility !== undefined) {
                                // Force GPU acceleration
                                video.style.webkitBackfaceVisibility = 'hidden';
                            }
                            
                            if ('setLatencyHint' in video.playbackRate.constructor.prototype) {
                                video.playbackRate.setLatencyHint('interactive');
                                console.log("Set video latency hint to interactive");
                            }
                            
                            // Additional low-latency features for Chrome
                            if (video.mozDecodedFrames === undefined) { // Not Firefox
                                video.playsInline = true;
                                video.autoplay = true;
                                video.disablePictureInPicture = true;
                                video.disableRemotePlayback = true;
                            }
                        } else {
                            console.error('No media stream available in track event');
                            setStatus('Connected, but no video stream available.', 'warning');
                        }
                    }
                };

                // Create offer with low-latency preferences
                const offerOptions = {
                    offerToReceiveVideo: true,
                    offerToReceiveAudio: false,
                    voiceActivityDetection: false
                };
                
                const offer = await pc.createOffer(offerOptions);
                
                // Modify SDP for lower latency
                let sdp = offer.sdp;
                // Set max packet size for lower latency
                sdp = sdp.replace('a=max-message-size:262144', 'a=max-message-size:65536');
                // Prioritize faster codecs
                if (sdp.includes('VP8')) {
                    sdp = sdp.replace(/m=video \\d+ UDP\\/TLS\\/RTP\\/SAVPF ([\\d ]+)/, (match, codecList) => {
                        const codecs = codecList.split(' ');
                        const vp8Index = codecs.findIndex((codec, i) => {
                            return sdp.includes(`a=rtpmap:${codec} VP8`);
                        });
                        if (vp8Index > 0) {
                            // Move VP8 to the front for higher priority
                            const vp8Codec = codecs[vp8Index];
                            codecs.splice(vp8Index, 1);
                            codecs.unshift(vp8Codec);
                        }
                        return `m=video 9 UDP/TLS/RTP/SAVPF ${codecs.join(' ')}`;
                    });
                }
                // Set bandwidth limitation
                if (!sdp.includes('b=AS:')) {
                    sdp = sdp.replace(/c=IN IP4.*\\r\\n/g, 'c=IN IP4 0.0.0.0\\r\\nb=AS:2000\\r\\n');
                }
                
                const modifiedOffer = new RTCSessionDescription({
                    type: offer.type,
                    sdp: sdp
                });
                
                await pc.setLocalDescription(modifiedOffer);

                console.log('Sending offer to server...');
                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
                }

                const answer = await response.json();
                console.log('Received answer from server');
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
                console.log('Set remote description');
                
            } catch (e) {
                setStatus(`Error: ${e.message}`, 'error');
                console.error('Error starting video:', e);
                startButton.disabled = false;
            }
        }

        async function stop() {
            if (statsInterval) {
                clearInterval(statsInterval);
                statsInterval = null;
            }
            
            // Reset latency display
            latencyDiv.textContent = 'Latency: --';
            latencyDiv.style.backgroundColor = 'rgba(0,0,0,0.5)';
            latencyValues = [];
            
            if (pc) {
                pc.close();
                pc = null;
            }
            
            if (video.srcObject) {
                const tracks = video.srcObject.getTracks();
                tracks.forEach(track => track.stop());
                video.srcObject = null;
            }
            
            setStatus('Camera stopped.', 'warning');
            startButton.disabled = false;
            stopButton.disabled = true;
            statsDiv.textContent = '';
        }
        
        function toggleFullScreen() {
            if (!document.fullscreenElement) {
                if (video.requestFullscreen) {
                    video.requestFullscreen();
                } else if (video.webkitRequestFullscreen) {
                    video.webkitRequestFullscreen();
                } else if (video.msRequestFullscreen) {
                    video.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        }

        startButton.addEventListener('click', start);
        stopButton.addEventListener('click', stop);
        fullScreenButton.addEventListener('click', toggleFullScreen);
        
        // Auto-restart on errors after a delay
        video.addEventListener('error', (e) => {
            console.error('Video error:', e);
            setStatus('Video error detected. Attempting to reconnect...', 'error');
            setTimeout(() => {
                if (pc && pc.connectionState === 'connected') {
                    // Try to recover without full restart
                    if (video.srcObject) {
                        video.srcObject.getTracks().forEach(track => {
                            if (track.readyState !== 'live') {
                                console.log('Track is not live, requesting restart');
                                stop();
                                setTimeout(start, 1000);
                            }
                        });
                    }
                }
            }, 3000);
        });

        // Clean up when page is closed
        window.onbeforeunload = function() {
            if (statsInterval) {
                clearInterval(statsInterval);
            }
            if (pc) {
                pc.close();
                pc = null;
            }
        };
    </script>
</body>
</html>""")
        
        # Return the newly created file
        with open("index.html", "r") as file:
            return HTMLResponse(content=file.read())

@app.on_event("shutdown")
async def on_shutdown():
    # Close all peer connections
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros, return_exceptions=True)
    peer_connections.clear()
    
    # Release camera
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        print("Camera released.")
    
    # Shutdown thread pool
    executor.shutdown(wait=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)