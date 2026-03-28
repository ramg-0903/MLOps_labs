## Overview

This project demonstrates how to build, containerize, and deploy a simple Flask-based **Unit Converter API** using Docker and Google Cloud Run. The application converts values between different units (e.g., centimeters to meters) and is deployed as a scalable, serverless web service.

---

## Steps

### 1. Set Up Google Cloud Project

* Created a Google Cloud project
* Enabled necessary APIs (Cloud Run, Artifact Registry, etc.)

### 2. Containerize the Application

* Wrote the Python Flask application for unit conversion
* Created a Dockerfile to containerize the app
* Built the Docker image locally

### 3. Push Image to Container Registry

* Tagged the Docker image
* Pushed the image to Google Artifact Registry

### 4. Deploy to Google Cloud Run

* Created a Cloud Run service
* Configured region, authentication, and container settings
* Deployed the containerized application

### 5. Monitor the Application

* Used Cloud Run logs and metrics to monitor performance and requests

---

## What I Changed

The original "Hello World" Flask app was modified into a **Unit Converter API** that performs conversions between different measurement units via API endpoints.

Instead of basic arithmetic operations like a calculator, this application:

* Accepts a value and unit types as input
* Converts between units (e.g., cm, m, km, mm)
* Returns the result in JSON format

---

## Example Usage

### Request

```
/convert?value=100&from_unit=cm&to_unit=m
```

### Response

```json
{
  "input_value": 100,
  "from_unit": "cm",
  "to_unit": "m",
  "converted_value": 1.0
}
```
