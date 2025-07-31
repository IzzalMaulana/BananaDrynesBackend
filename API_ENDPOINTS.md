# API Endpoints - BananaDrynes Backend

## Base URL
```
http://your-domain.com
```

## Endpoints

### 1. Health Check
```
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "database": "OK",
  "xgboost_model": "OK", 
  "vit_model": "OK"
}
```

### 2. Predict Image
```
POST /predict
```
**Request:** Form data with image file
**Response:**
```json
{
  "classification": "Kering",
  "accuracy": 95.2,
  "drynessLevel": 2,
  "filename": "image.jpg"
}
```

### 3. Get History
```
GET /history
```
**Response:**
```json
[
  {
    "id": 1,
    "filename": "image.jpg",
    "classification": "Kering",
    "accuracy": 95.2,
    "drynessLevel": 2,
    "created_at": "2024-01-15 14:30:25"
  }
]
```

### 4. Delete Single History
```
DELETE /history/{id}
```
**Parameters:**
- `id` (integer): ID history yang akan dihapus

**Response Success:**
```json
{
  "message": "History deleted successfully"
}
```

**Response Error (404):**
```json
{
  "error": "History record not found"
}
```

### 5. Clear All History
```
DELETE /history/clear
```
**Response Success:**
```json
{
  "message": "All history cleared successfully",
  "deleted_records": 10,
  "deleted_files": 8
}
```

### 6. Get Uploaded File
```
GET /uploads/{filename}
```
**Parameters:**
- `filename` (string): Nama file yang akan diambil

**Response:** File binary

## Error Responses

### 500 Internal Server Error
```json
{
  "error": "Prediction failed: Model not available"
}
```

### 400 Bad Request
```json
{
  "error": "No image uploaded"
}
```

## Frontend Integration

### Untuk Delete Single History:
```javascript
const deleteHistory = async (id) => {
  try {
    const response = await fetch(`/history/${id}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (response.ok) {
      // Refresh history list
      fetchHistory();
    } else {
      const error = await response.json();
      alert('Error: ' + error.error);
    }
  } catch (error) {
    console.error('Error deleting history:', error);
    alert('Failed to delete history');
  }
};
```

### Untuk Clear All History:
```javascript
const clearAllHistory = async () => {
  if (confirm('Are you sure you want to delete all history?')) {
    try {
      const response = await fetch('/history/clear', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Deleted ${result.deleted_records} records and ${result.deleted_files} files`);
        // Refresh history list
        fetchHistory();
      } else {
        const error = await response.json();
        alert('Error: ' + error.error);
      }
    } catch (error) {
      console.error('Error clearing history:', error);
      alert('Failed to clear history');
    }
  }
};
```

## Notes

1. **File Cleanup**: Endpoint delete akan otomatis menghapus file gambar dari server
2. **Error Handling**: Semua endpoint memiliki error handling yang proper
3. **Database**: Menggunakan MySQL dengan tabel `history`
4. **CORS**: Sudah dikonfigurasi untuk frontend
5. **Timeout**: Endpoint predict memiliki timeout 300 detik untuk model AI 