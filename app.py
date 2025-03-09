from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import json
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import cv2
from typing import List, Dict, Any

# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier('static/models/haarcascade_frontalface_default.xml')
import face_recognition

import io

# Initialize FastAPI app
app = FastAPI(title="Face Recognition Attendance System")

# Create necessary directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/sounds", exist_ok=True)
os.makedirs("static/models", exist_ok=True)
os.makedirs("Dataset", exist_ok=True)
os.makedirs("AttendanceData", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables
KNOWN_FACES_FILE = "known_faces.json"

# Helper functions
def initialize_known_faces_file():
    """Initialize known_faces.json file if it doesn't exist"""
    if not os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "w") as f:
            json.dump([], f)

def load_known_faces():
    """Load known faces from JSON file"""
    if not os.path.exists(KNOWN_FACES_FILE):
        initialize_known_faces_file()
        
    with open(KNOWN_FACES_FILE, "r") as f:
        return json.load(f)

def save_known_faces(faces_data):
    """Save known faces to JSON file"""
    with open(KNOWN_FACES_FILE, "w") as f:
        json.dump(faces_data, f, indent=2)

def base64_to_image(base64_string):
    """Convert base64 image string to OpenCV image"""
    # Remove the data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def get_face_encoding(img):
    """Get face encoding from image using Haar Cascade"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Get the first face detected
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    # Get face encoding (you can use any encoding method here)
    face_encoding = face_recognition.face_encodings(face_img)

    if not face_encoding:
        return None

    return face_encoding[0].tolist()

def get_attendance_file_path():
    """Get the path to the current month's attendance file"""
    now = datetime.now()
    month_year = now.strftime("%B-%Y")  # e.g., "March-2025"
    file_name = f"{month_year}.xlsx"
    return os.path.join("AttendanceData", file_name)

def initialize_attendance_file(file_path):
    """Initialize the attendance Excel file if it doesn't exist"""
    if not os.path.exists(file_path):
        # Create a new DataFrame with the required columns
        df = pd.DataFrame(columns=["Date", "Name", "EntryTime", "ExitTime"])
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to Excel
        df.to_excel(file_path, index=False)
    
    return file_path

def log_attendance_to_excel(name, entry_time=None, exit_time=None):
    """Log attendance to Excel file"""
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get the attendance file path
    file_path = get_attendance_file_path()
    
    # Initialize the file if it doesn't exist
    initialize_attendance_file(file_path)
    
    # Read the existing Excel file
    df = pd.read_excel(file_path)
    
    # Check if there's already an entry for this person today
    today_record = df[(df["Date"] == today) & (df["Name"] == name)]
    
    if len(today_record) == 0:
        # No record for today, create a new one with entry time
        new_record = {
            "Date": today,
            "Name": name,
            "EntryTime": entry_time or datetime.now().strftime("%H:%M:%S"),
            "ExitTime": exit_time or ""
        }
        df = df.append(new_record, ignore_index=True)
    else:
        # Update existing record with exit time
        df.loc[(df["Date"] == today) & (df["Name"] == name), "ExitTime"] = exit_time or datetime.now().strftime("%H:%M:%S")
    
    # Save the updated DataFrame
    df.to_excel(file_path, index=False)

# Initialize the system
initialize_known_faces_file()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the index.html page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/update", response_class=HTMLResponse)
async def get_update(request: Request):
    """Serve the update.html page"""
    return templates.TemplateResponse("update.html", {"request": request})

@app.get("/api/known-faces")
async def get_known_faces():
    """Get all known faces"""
    return load_known_faces()

@app.post("/api/register-employee")
async def register_employee(request: Request):
    """Register a new employee with face images"""
    try:
        # Parse request body
        data = await request.json()
        name = data.get("name")
        images = data.get("images", [])
        
        if not name or not images:
            raise HTTPException(status_code=400, detail="Name and images are required")
        
        # Create employee directory in Dataset folder
        employee_dir = os.path.join("Dataset", name)
        os.makedirs(employee_dir, exist_ok=True)
        
        # Process and save face images
        face_encodings = []
        
        for i, img_base64 in enumerate(images):
            # Convert base64 to image
            img = base64_to_image(img_base64)
            
            if img is None:
                continue
            
            # Save image to dataset folder
            img_path = os.path.join(employee_dir, f"{name}_{i}.jpg")
            cv2.imwrite(img_path, img)
            
            # Get face encoding
            face_encoding = get_face_encoding(img)
            
            if face_encoding:
                face_encodings.append(face_encoding)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No faces detected in the provided images")
        
        # Calculate average face encoding
        avg_encoding = np.mean(face_encodings, axis=0).tolist()
        
        # Add to known faces
        known_faces = load_known_faces()
        
        # Check if employee already exists
        employee_exists = False
        for face in known_faces:
            if face["name"] == name:
                face["descriptor"] = avg_encoding
                employee_exists = True
                break
        
        if not employee_exists:
            known_faces.append({
                "name": name,
                "descriptor": avg_encoding
            })
        
        # Save updated known faces
        save_known_faces(known_faces)
        
        return {"success": True, "message": f"Employee {name} registered successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log-attendance")
async def log_attendance(request: Request):
    """Log attendance for an employee"""
    try:
        # Parse request body
        data = await request.json()
        name = data.get("name")
        date = data.get("date")
        entry_time = data.get("entryTime")
        exit_time = data.get("exitTime")
        
        # Debugging logs
        print(f"Logging attendance for: {name}, Entry Time: {entry_time}, Exit Time: {exit_time}")
        
        if not name:
            raise HTTPException(status_code=400, detail="Employee name is required")
        
        # Determine if this is an entry or exit
        if entry_time and not exit_time:
            # This is an entry
            log_attendance_to_excel(name, entry_time=entry_time)
        elif entry_time and exit_time:
            # This is an exit
            log_attendance_to_excel(name, entry_time=entry_time, exit_time=exit_time)
        
        return {"success": True, "message": "Attendance logged successfully"}
    
    except Exception as e:
        print(f"Error logging attendance: {str(e)}")  # Debugging log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-status")
async def get_system_status():
    """Get the system status including number of registered employees and recent attendance"""
    try:
        # Get known faces count
        known_faces = load_known_faces()
        employee_count = len(known_faces)
        
        # Get today's attendance count
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = get_attendance_file_path()
        
        if os.path.exists(attendance_file):
            df = pd.read_excel(attendance_file)
            today_attendance = len(df[df["Date"] == today])
        else:
            today_attendance = 0
        
        # Calculate system uptime (mock data)
        uptime = "24h"  # This would typically be calculated from system start time
        
        return {
            "employeeCount": employee_count,
            "todayAttendance": today_attendance,
            "systemUptime": uptime,
            "status": "Running"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance-data")
async def get_attendance_data(month: str = None, year: str = None):
    """Get attendance data for a specific month and year"""
    try:
        # If month and year are not provided, use current month and year
        if not month or not year:
            now = datetime.now()
            month = now.strftime("%B")  # Full month name
            year = now.strftime("%Y")
        
        # Get the attendance file path
        file_name = f"{month}-{year}.xlsx"
        file_path = os.path.join("AttendanceData", file_name)
        
        if not os.path.exists(file_path):
            return {"data": [], "message": "No attendance data for the specified month"}
        
        # Read the attendance data
        df = pd.read_excel(file_path)
        
        # Convert DataFrame to list of dictionaries
        attendance_data = df.to_dict(orient="records")
        
        return {"data": attendance_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/employee-list")
async def get_employee_list():
    """Get a list of all registered employees"""
    try:
        known_faces = load_known_faces()
        employees = [face["name"] for face in known_faces]
        return {"employees": employees}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete-employee/{name}")
async def delete_employee(name: str):
    """Delete an employee from the system"""
    try:
        # Load known faces
        known_faces = load_known_faces()
        
        # Filter out the employee to delete
        known_faces = [face for face in known_faces if face["name"] != name]
        
        # Save updated known faces
        save_known_faces(known_faces)
        
        # Delete employee directory from Dataset folder
        employee_dir = os.path.join("Dataset", name)
        if os.path.exists(employee_dir):
            shutil.rmtree(employee_dir)
        
        return {"success": True, "message": f"Employee {name} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}")
async def serve_model(model_name: str):
    """Serve face-api.js model files"""
    model_path = os.path.join("static", "models", model_name)
    if os.path.exists(model_path):
        return FileResponse(model_path)
    else:
        raise HTTPException(status_code=404, detail=f"Model file {model_name} not found")

@app.get("/sounds/{sound_name}")
async def serve_sound(sound_name: str):
    """Serve sound files"""
    sound_path = os.path.join("static", "sounds", sound_name)
    if os.path.exists(sound_path):
        return FileResponse(sound_path)
    else:
        raise HTTPException(status_code=404, detail=f"Sound file {sound_name} not found")

@app.get("/api/monthly-reports")
async def get_monthly_reports():
    """Get a list of available monthly attendance reports"""
    try:
        # Get all Excel files in the AttendanceData folder
        reports = []
        for file in os.listdir("AttendanceData"):
            if file.endswith(".xlsx"):
                month_year = file.replace(".xlsx", "")
                reports.append(month_year)
        
        return {"reports": reports}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-report/{report_name}")
async def download_report(report_name: str):
    """Download a specific monthly attendance report"""
    try:
        file_path = os.path.join("AttendanceData", f"{report_name}.xlsx")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=file_path,
            filename=f"{report_name}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)