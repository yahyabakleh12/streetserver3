# StreetServer3

StreetServer2 is a FastAPI application that receives parking reports from network cameras and stores them as JSON files. When a vehicle is detected occupying a spot, a snapshot is processed through a YOLO-based OCR pipeline to read the license plate. Tickets are then created in the database and optionally synchronized with the Parkonic API.

## Requirements

Python 3.10 or later is recommended. Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configuration

The application reads configuration from environment variables. Set the
following variables before running the server:

- `DATABASE_URL` – MySQL connection string using the PyMySQL driver
  (`mysql+pymysql`). This variable is required.
- `OCR_TOKEN` – token for the OCR service. This variable is required.
 - `YOLO_MODEL_PATH` – path to the YOLOv11x vehicle detection model (can also be changed in
   `config.py`).
- `REAL_ESRGAN_MODEL_PATH` – path to the RealESRGAN weights used for plate image enhancement.
- `CORS_ORIGINS`  – comma-separated list of origins allowed to access the API.
  Use `*` to allow requests from any host.
- `RABBITMQ_URL` – URL for the RabbitMQ message broker used by Dramatiq.
  This variable is required for background processing.

Camera credentials and the Parkonic API token are now stored per location in the
`locations` table instead of being global environment variables.

## Running the server

Make sure MySQL is running and the tables defined in `models.py` exist. Then start the API with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
All camera triggers are now served from the same application, so only one process is required.

### Docker Compose stack

For local development the repository ships with a `docker-compose.yml` that provisions
TimescaleDB, MinIO (plus a setup job that bootstraps an object-storage bucket) and the API
service. Build and start the stack with:

```bash
docker compose up --build
```

Default credentials are stored in the compose file and can be overridden with environment
variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | Database credentials used by TimescaleDB and the API | `street` |
| `POSTGRES_DB` | Database name created at startup | `streetserver` |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | MinIO access and secret key | `minioadmin` / `minioadminsecret` |
| `MINIO_BUCKET_NAME` | Bucket created by the helper job | `streetserver` |

Once the containers are running the API is available at http://localhost:8000, TimescaleDB at
localhost:5432 and MinIO at http://localhost:9000 (with the console at http://localhost:9001).

### Background processing with Dramatiq

Set `RABBITMQ_URL` to a persistent RabbitMQ instance. The application will
refuse to start if this variable is missing.

Incoming `/post` events are queued in Dramatiq. Camera events are distributed
across fixed shards so that tasks for a given camera are processed in order.

Start a worker for every queue shard defined in `tasks.py`. The helper script
`start_workers.py` launches workers for all configured shards:

```bash
./start_workers.py
```

To start a single worker manually for shard 0:

```bash
dramatiq tasks -Q post-shard-0,plate-shard-0 --processes 1
```

Repeat for each shard index if launching workers manually. Entry and exit events
for the same camera must target the same shard to preserve ordering. Clip
requests are processed inline using a thread pool rather than FastAPI's
`BackgroundTasks`.

The API exposes a `/post` endpoint that accepts JSON payloads describing parking events. This endpoint is intended for camera devices and does **not** require authentication.
When a device reports an exit (`occupancy` set to `0`), the application now grabs the latest frame from the camera and checks the spot using the plate detector. If a plate is still visible the ticket remains open and the endpoint responds that the spot is still occupied.

An additional `/ocr-image` endpoint allows uploading a picture of a vehicle. The service crops the detected license plate, runs it through the OCR pipeline and returns the parsed result as JSON. This endpoint does not require authentication and expects a multipart form field named `image`.

### Authentication

Most endpoints are protected using bearer tokens. First create a user in the `users`
table and obtain a token:

```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=YOUR_USER&password=YOUR_PASS"
```

The JSON response includes the access token and token type. The JWT payload now
also contains a `roles` field listing the names of the user's roles.
Tokens are valid for 24 hours by default.

Use the returned token in the `Authorization` header when calling other
endpoints:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/tickets
```

### Roles and permissions

StreetServer2 implements role-based access control (RBAC).
The initial SQL dump defines a `superadmin` role linked to user id 1, granting full access to all permissions. Each user may belong
to one or more roles. Roles are assigned permissions which gate access to the
management endpoints. The application defines three permission names used by the
API:

- `manage_users`
- `manage_roles`
- `manage_permissions`

The following endpoints are available for RBAC management (all require an
authorized user with the appropriate permission):

- `/users` – create, list, retrieve, update and delete users
- `/roles` – create, list, retrieve, update and delete roles
- `/permissions` – create, list, retrieve, update and delete permissions

Example workflow to set up a new user:

```bash
# create a role that can manage users
curl -X POST http://localhost:8000/roles \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "admin", "permission_ids": [1,2,3]}'

# create the user and assign the role by id
curl -X POST http://localhost:8000/users \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret", "role_ids": [1]}'

# obtain a login token for the new account
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=alice&password=secret"
```

### Listing tickets

Use the `/tickets` endpoint to retrieve issued tickets. It supports pagination
and basic searching by license plate number. Additional filters allow selecting
tickets for a specific camera, spot and license plate details as well as a time
range for the entry timestamp.

Query parameters:

- `page` – page number starting from 1 (default `1`)
- `page_size` – number of items per page (default `50`)
- `search` – partial plate number to filter by
- `camera_id` – only return tickets for this camera
- `spot_number` – only return tickets for this spot number
- `plate_number` – filter by the full plate number
- `plate_code` – filter by the plate code
- `plate_city` – filter by the emirate/city on the plate
- `entry_start` – ISO timestamp for the earliest entry time
- `entry_end` – ISO timestamp for the latest entry time
- `sort_by` – field to sort on (`id`, `entry_time`, etc.)
- `sort_order` – `asc` or `desc` (default `desc`)

Example:

```bash
curl "http://localhost:8000/tickets?page=1&page_size=20&plate_number=ABC123&entry_start=2024-01-01T00:00:00&entry_end=2024-01-31T23:59:59"
```

### Downloading ticket media

Use `/tickets/{id}/image` to download the entry snapshot for a ticket and
`/tickets/{id}/video` to download the exit clip if available.

```bash
curl -O http://localhost:8000/tickets/123/image
curl -O http://localhost:8000/tickets/123/video
```

### Listing manual reviews

Use the `/manual-reviews` endpoint to retrieve events that require human
verification. The endpoint supports simple pagination.

Query parameters:

- `status` – review status to filter by (`PENDING` or `RESOLVED`, default `PENDING`)
- `page` – page number starting from 1 (default `1`)
- `page_size` – number of items per page (default `50`)

Example:

```bash
curl "http://localhost:8000/manual-reviews?page=1&page_size=20"
```

### Retrieving a manual review

Fetch details for a specific review by ID using `/manual-reviews/{id}`.

```bash
curl "http://localhost:8000/manual-reviews/123"
```

### Correcting a manual review

Submit updated plate information when a review has been manually verified using
`/manual-reviews/{review_id}/correct`.

Required JSON fields:

- `plate_number` – license plate number
- `plate_code` – plate code
- `plate_city` – issuing city
- `confidence` – confidence value as an integer

Example:

```bash
curl -X POST http://localhost:8000/manual-reviews/123/correct \
  -H "Content-Type: application/json" \
  -d '{"plate_number": "ABC123", "plate_code": "12", "plate_city": "DXB", "confidence": 95}'
```

### External manual review correction

Clients can also submit corrections without authentication using `/external-corrections`.

Required JSON fields:

- `review_id` – ID of the manual review
- `plate_number` – license plate number
- `plate_code` – plate code
- `plate_city` – issuing city
- `image_base64` – base64-encoded plate image

If the linked ticket is still open only the entry event is sent to the portal.
Closed tickets trigger both entry and exit events before marking the review as resolved.

Example:

```bash
curl -X POST http://localhost:8000/external-corrections \
  -H "Content-Type: application/json" \
  -d '{"review_id": 123, "plate_number": "ABC123", "plate_code": "12", "plate_city": "DXB", "image_base64": "..."}'
```

### Dismissing a manual review

To dismiss a review without changing the ticket, use
`/manual-reviews/{review_id}/dismiss`.

```bash
curl -X POST http://localhost:8000/manual-reviews/123/dismiss
```

### Sending pending review videos

Trigger upload of all unresolved reviews that already have a clip using
`/manual-reviews/send-videos`.

Query parameters:

- `server_base` – base URL used to construct the download link in the payload
  (default `http://localhost:8000`).

Example:

```bash
curl -X POST http://localhost:8000/manual-reviews/send-videos
```

### Deleting reviews without videos

Remove manual review entries that lack a video file using
`/manual-reviews/delete-missing-videos`.

```bash
curl -X POST http://localhost:8000/manual-reviews/delete-missing-videos
```

### Location statistics

Retrieve counts of zones, poles, cameras, tickets and manual reviews grouped by
location using `/location-stats`.

```bash
curl http://localhost:8000/location-stats -H "Authorization: Bearer <token>"
```

### Camera occupancy

List cameras with their currently occupied parking spots using `/camera-occupancy`.
Append the location ID to the path to limit results to that location.

```bash
curl http://localhost:8000/camera-occupancy -H "Authorization: Bearer <token>"
curl http://localhost:8000/camera-occupancy/1 \
  -H "Authorization: Bearer <token>"
```

### Managing parking spots

Create a new spot using `/spots`, retrieve a spot with `/spots/{id}`, and
list spots for a camera with `/cameras/{id}/spots`.

```bash
curl -X POST http://localhost:8000/spots \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"camera_id": 1, "spot_number": 1,
       "p1_x": 0,  "p1_y": 0,
       "p2_x": 100, "p2_y": 0,
       "p3_x": 100, "p3_y": 50,
       "p4_x": 0,  "p4_y": 50,
       "status": 0}'
```

### Fetching camera frames

Grab the latest snapshot from a camera using `/cameras/{id}/frame`.

```bash
curl http://localhost:8000/cameras/1/frame \
  -H "Authorization: Bearer <token>" -o frame.jpg
```

## License

This project is released under the terms of the MIT License. See [LICENSE](LICENSE) for the full text.
# streetserver3
