# Docker Setup for Naive Bayes Classifier API

This guide covers Docker deployment for the FastAPI backend.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

### Development Environment

```bash
# Build and start all services
make build
make up

# Or using docker-compose directly
docker-compose up -d

# Access the application
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Adminer (DB UI): http://localhost:8080
```

### Production Environment

```bash
# Build and start production containers
make prod-build
make prod-up
```

## Docker Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| API | `naive_bayes_api` | 8000 | FastAPI application |
| MySQL | `naive_bayes_mysql` | 3306 | MySQL database |
| Adminer | `naive_bayes_adminer` | 8080 | Database management UI (optional) |
| Nginx | `naive_bayes_nginx` | 80/443 | Reverse proxy (production only) |

## Environment Variables

Create a `.env` file from the template:

```bash
cp .env.docker .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8000 | API port |
| `DB_HOST` | mysql | Database host (service name) |
| `DB_USER` | nb_user | Database user |
| `DB_PASSWORD` | nb_password | Database password |
| `DB_NAME` | naive_bayes | Database name |
| `SECRET_KEY` | - | JWT secret key (change in production!) |

## Default Credentials

### Admin User
- Email: `admin@naivebayes.local`
- Password: `admin123`

### Database
- Host: `localhost:3306` (from host) or `mysql:3306` (from containers)
- User: `nb_user`
- Password: `nb_password`
- Database: `naive_bayes`

## Makefile Commands

```bash
# Development
make build       # Build containers
make up          # Start services
make down        # Stop services
make restart     # Restart services
make logs        # View all logs
make logs-api    # View API logs
make logs-db     # View MySQL logs
make shell       # Open shell in API container
make test        # Run tests
make clean       # Remove all containers and volumes

# Production
make prod-build  # Build production containers
make prod-up     # Start production services
make prod-down   # Stop production services

# Database
make db-migrate  # Initialize database tables
make db-shell    # Open MySQL shell
```

## Development Workflow

### Hot-Reload Development

```bash
# Use development docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This mounts your source code and enables hot-reload.

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
docker-compose exec api pytest tests/test_auth.py -v

# Run with coverage
docker-compose exec api pytest --cov=app --cov-report=html
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f mysql

# Last 100 lines
docker-compose logs --tail=100 api
```

## Production Deployment

### Building Production Image

```bash
docker build -t naive-bayes-api:latest .
```

### Running with Docker Compose

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Manual Docker Run

```bash
docker run -d \
  --name naive-bayes-api \
  -p 8000:8000 \
  -e DB_HOST=your-db-host \
  -e DB_PASSWORD=your-db-password \
  -e SECRET_KEY=your-secret-key \
  -v $(pwd)/data:/app/data \
  naive-bayes-api:latest
```

## Health Checks

The API includes a health check endpoint:

```bash
curl http://localhost:8000/health
```

Docker health check status:

```bash
docker ps
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs api

# Check container status
docker-compose ps
```

### Database connection issues

```bash
# Verify MySQL is running
docker-compose logs mysql

# Test database connection
docker-compose exec api python -c "from app.core.database import get_db; print('OK')"
```

### Volume issues

```bash
# Remove volumes and start fresh
make clean
make up
```

### Rebuild after code changes

```bash
# Rebuild specific service
docker-compose build api

# Rebuild all services
docker-compose build
```

## Backup and Restore

### Backup Database

```bash
docker-compose exec mysql mysqldump -u nb_user -pnb_password naive_bayes > backup.sql
```

### Restore Database

```bash
docker-compose exec -T mysql mysql -u nb_user -pnb_password naive_bayes < backup.sql
```

### Backup Data Volumes

```bash
docker run --rm \
  -v naive_bayes_backend-new_mysql_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/mysql-backup.tar.gz -C /data .
```

## Security Notes

1. **Change default passwords** in production
2. **Use strong SECRET_KEY** (generate with `openssl rand -hex 32`)
3. **Disable Adminer** in production (remove `tools` profile)
4. **Use HTTPS** in production (configure SSL in nginx)
5. **Set up firewall rules** to restrict access
6. **Regular updates** of base images

## Monitoring

### Container Stats

```bash
docker stats
```

### Resource Usage

```bash
docker-compose top
```

### Logs Analysis

```bash
# Error logs
docker-compose logs api | grep ERROR

# Access logs (if nginx enabled)
docker-compose logs nginx | grep -v "health"
```
