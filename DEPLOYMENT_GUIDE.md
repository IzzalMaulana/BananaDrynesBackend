# Panduan Deployment Manual - BananaDrynes Backend

## Setup yang Benar (Manual Deployment)

### 1. Update Kode Backend

```bash
# Masuk ke direktori project
cd /path/to/your/BananaDrynesBackend

# Backup kode lama
cp main.py main.py.backup

# Update dengan kode baru yang sudah diperbaiki
# (Upload semua file yang sudah diperbaiki)
```

### 2. Periksa Database

```bash
# Masuk ke MySQL
mysql -u root -p

# Buat database dan user jika belum ada
CREATE DATABASE IF NOT EXISTS banana_db;
CREATE USER IF NOT EXISTS 'banana_user'@'localhost' IDENTIFIED BY 'password_anda';
GRANT ALL PRIVILEGES ON banana_db.* TO 'banana_user'@'localhost';
FLUSH PRIVILEGES;

# Buat tabel history
USE banana_db;
CREATE TABLE IF NOT EXISTS history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    classification VARCHAR(100) NOT NULL,
    accuracy DECIMAL(5,2) NOT NULL,
    drynessLevel INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# Keluar dari MySQL
EXIT;
```

### 3. Update Systemd Service

```bash
# Edit file service
sudo nano /etc/systemd/system/bananadrynes.service
```

Isi dengan konfigurasi yang benar (sesuaikan path):

```ini
[Unit]
Description=BananaDrynes Backend
After=network.target mysql.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/your/BananaDrynesBackend
Environment=PATH=/path/to/your/BananaDrynesBackend/venv/bin
ExecStart=/path/to/your/BananaDrynesBackend/venv/bin/gunicorn --config gunicorn.conf.py main:app
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Memory limits
MemoryMax=3G
MemoryHigh=2.5G

# Timeout settings
TimeoutStartSec=300
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

### 4. Update Nginx Configuration

```bash
# Edit file nginx
sudo nano /etc/nginx/sites-available/bananadrynes
```

Isi dengan konfigurasi yang benar:

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    # Upload size limit
    client_max_body_size 10M;

    # Backend API
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Static files
    location /uploads/ {
        alias /path/to/your/BananaDrynesBackend/uploads/;
    }
}
```

### 5. Install Dependencies

```bash
# Aktifkan virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Buat direktori uploads
mkdir -p uploads
chmod 755 uploads
```

### 6. Restart Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart bananadrynes

# Restart nginx
sudo systemctl restart nginx

# Cek status
sudo systemctl status bananadrynes
sudo systemctl status nginx
```

### 7. Test Aplikasi

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test dari luar (ganti dengan IP/domain Anda)
curl http://your-domain.com/health
```

### 8. Monitor Logs

```bash
# Monitor service logs
sudo journalctl -u bananadrynes -f

# Monitor nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

## Troubleshooting

### Jika Error 502:

1. **Cek service status:**
   ```bash
   sudo systemctl status bananadrynes
   ```

2. **Cek memory usage:**
   ```bash
   free -h
   ```

3. **Cek port 8000:**
   ```bash
   netstat -tlnp | grep 8000
   ```

4. **Restart service:**
   ```bash
   sudo systemctl restart bananadrynes
   ```

### Jika History Gagal:

1. **Test database connection:**
   ```bash
   mysql -u banana_user -p banana_db -e "SELECT 1"
   ```

2. **Cek tabel history:**
   ```bash
   mysql -u banana_user -p banana_db -e "SHOW TABLES"
   ```

3. **Restart MySQL:**
   ```bash
   sudo systemctl restart mysql
   ```

## File yang Tidak Perlu

- `nixpacks.toml` - Tidak digunakan dalam setup manual
- `vercel.json` - Untuk Vercel deployment

## File yang Penting

- `main.py` - Aplikasi utama
- `gunicorn.conf.py` - Konfigurasi Gunicorn
- `requirements.txt` - Dependencies
- `bananadrynes.service` - Systemd service
- Nginx configuration 