# API Contract Specification

All backend endpoints are hosted under prefix `/api/background-remover` on FastAPI server (`http://127.0.0.1:8000`).

---

## Endpoint 1: Process & Remove Background

- **HTTP Method**: `POST`
- **Path**: `/api/background-remover/remove`
- **Content-Type**: `multipart/form-data`
- **Purpose**: Process uploaded image through BiRefNet General FP16 ONNX and classical CV refinement. Returns transparent base64 PNG data URL, dimensions, and stage timings.

### Form Fields
| Field Name | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `file` | `UploadFile` | *Required* | Max 60 MB, JPG/PNG/WEBP/AVIF/BMP/TIFF | Target image binary stream |
| `sensitivity` | `float` | `10.0` | `0.0` to `100.0` (%) | Trimap background sensitivity threshold |
| `edge_softness` | `float` | `50.0` | `0.0` to `100.0` (%) | Transition band alpha feathering softness |
| `defringe_strength` | `float` | `50.0` | `0.0` to `100.0` (%) | Boundary color decontamination strength |
| `diagnostics` | `bool` | `False` | `True` or `False` | Include detailed stage timing breakdown |

### Response (`HTTP 200 OK`)
```json
{
  "success": true,
  "requestId": "BG-20260918-001",
  "engine": "BackgroundRemovalEngine",
  "model": "BiRefNet General FP16 (ONNX)",
  "width": 1408,
  "height": 768,
  "processingTimeMs": 2040.52,
  "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
  "diagnostics": {
    "stage_timings": {
      "decode_ms": 42.10,
      "inference_ms": 1820.30,
      "mask_quality_ms": 25.10,
      "trimap_ms": 30.50,
      "transparency_ms": 32.10,
      "alpha_refine_ms": 85.20,
      "defringe_ms": 95.10,
      "assembly_ms": 15.20,
      "encoding_ms": 912.40
    }
  }
}
```

### Error Responses
- **`HTTP 400 Bad Request`**: Uploaded file empty, format unsupported, or size exceeds 60 MB / 33.17M pixel limit.
- **`HTTP 429 Too Many Requests`**: Returned when `HEAVY_AI_LOCK` is held by another active inference request.
- **`HTTP 500 Internal Server Error`**: Unexpected runtime error during model inference or memory allocation.

---

## Endpoint 2: Composite Background

- **HTTP Method**: `POST`
- **Path**: `/api/background-remover/composite`
- **Content-Type**: `application/json`
- **Purpose**: Composite transparent base64 cutout with solid color, gradient, or transparent background and export as PNG, WEBP, or JPG.

### Request Body (`application/json`)
```json
{
  "image_base64": "data:image/png;base64,...",
  "bg_type": "solid",
  "color1": "#3B82F6",
  "color2": "#1E3A8A",
  "gradient_direction": "to-bottom",
  "export_format": "PNG",
  "quality": 90
}
```

### Response (`HTTP 200 OK`)
```json
{
  "success": true,
  "image_data": "data:image/png;base64,..."
}
```

---

## Endpoint 3: Export HD (Super-Resolution 2x / 4x)

- **HTTP Method**: `POST`
- **Path**: `/api/background-remover/export-hd`
- **Content-Type**: `application/json`
- **Purpose**: Upscale base64 cutout using Real-ESRGAN (2x or 4x) and YuNet face detail preservation.

### Request Body (`application/json`)
```json
{
  "image_base64": "data:image/png;base64,...",
  "export_quality": "hd2x",
  "export_format": "PNG",
  "bg_type": "transparent",
  "quality": 95
}
```

### Response (`HTTP 200 OK`)
```json
{
  "success": true,
  "image_data": "data:image/png;base64,...",
  "upscale_factor": 2,
  "new_width": 2816,
  "new_height": 1536
}
```

---

## Endpoint 4: System Status & Health Checks

- **`GET /api/background-remover/health`**: Returns status `online`, engine name, active device (`CUDA` / `CPU`), and privacy policy.
- **`GET /api/background-remover/status`**: Returns model manager state (`IDLE`, `READY`, `PROCESSING`), active providers, process RSS memory in MB, idle timer status, and cached model load times.
- **`GET /api/background-remover/debug/latest`**: Returns diagnostic file paths and timings for the most recent request.
