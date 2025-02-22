# Price Optimization API Documentation

## Base URL
```
http://localhost:8000
```

## Core Endpoints

### 1. Check Model Existence
Check if a trained model exists for a product.

**Endpoint:** `GET /models/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)

**Request Example:**
```bash
curl "http://localhost:8000/models/woolball"
```

**Response:**
```json
{
  "training_date": "2025-02-21T12:00:00",
  "data_range": {
    "start": "2024-10-01",
    "end": "2025-01-22"
  },
  "data_stats": {
    "total_records": 100,
    "price_records": 95,
    "sales_records": 95,
    "predicted_sales_records": 90
  },
  "training_config": {
    "exploration_rate": 1.0,
    "price_increase_bias": 0.2,
    "num_episodes": 1000
  },
  "performance": {
    // training metrics
  }
}
```

**Error Response (404):**
```json
{
  "detail": "Model not found for woolball"
}
```

### 2. Train Model
Train a new model for a product using historical data.

**Endpoint:** `POST /train/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)
- `file`: CSV file with historical data
- `config`: Training configuration (optional)
- `force`: Boolean flag to force retrain if model exists (optional, default: false)

**Request Example:**
```bash
# Train new model
curl -X POST "http://localhost:8000/train/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv" \
  -F 'config={
    "exploration_rate": 1.0,
    "price_increase_bias": 0.2,
    "num_episodes": 1000
  }'

# Force retrain existing model
curl -X POST "http://localhost:8000/train/woolball?force=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv"
```

**Success Response (200):**
```json
{
  "status": "success",
  "product_id": "woolball",
  "metadata": {
    "training_date": "2025-02-21T12:00:00",
    "data_range": {
      "start": "2024-10-01",
      "end": "2025-01-22"
    },
    "data_stats": {
      "total_records": 100,
      "price_records": 95,
      "sales_records": 95,
      "predicted_sales_records": 90
    },
    "training_config": {
      "exploration_rate": 1.0,
      "price_increase_bias": 0.2,
      "num_episodes": 1000
    },
    "performance": {
      // training metrics
    }
  }
}
```

**Error Response (409) - Model Exists:**
```json
{
  "detail": "Model already exists for woolball. Use force=true to retrain.",
  "model_path": "/path/to/model/checkpoint.pt",
  "metadata": {
    // existing model metadata
  }
}
```

**Error Response (500) - Training Failed:**
```json
{
  "detail": "Training failed: [error details]"
}
```

### 2. Generate Predictions
Get price predictions using trained model.

**Endpoint:** `POST /predict/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)
- `file`: CSV file with recent data
- `config`: Prediction configuration

**Request Example:**
```bash
curl -X POST "http://localhost:8000/predict/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/recent_data.csv" \
  -F 'config={
    "future_dates": ["2025-01-23", "2025-01-24"],
    "exploration_mode": false,
    "exploration_rate": 0.1
  }'
```

**Response:**
```json
[
  {
    "date": "2025-01-23",
    "recommended_price": 25.5,
    "predicted_sales": 150.0,
    "metrics": {
      "organic_conversion": 2.5,
      "ad_conversion": 1.8,
      "predicted_profit": 750.0
    },
    "exploration_info": {
      "exploration_bonus": 0.1,
      "exploration_std": 0.05,
      "price_vs_median": 1.02,
      "is_new_price_point": false
    }
  }
]
```

## Visualization Endpoints

### 1. Historical Analysis
Generate visualizations for historical data.

**Endpoint:** `POST /viz/visualize/historical/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)
- `file`: CSV file with historical data

**Request Example:**
```bash
curl -X POST "http://localhost:8000/viz/visualize/historical/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv"
```

**Response:**
```json
{
  "historical_analysis": {
    // Plotly figure data for price/sales time series
  },
  "price_sales_scatter": {
    // Plotly figure data for price vs sales scatter
  },
  "conversion_analysis": {
    // Plotly figure data for conversion rates
  }
}
```

### 2. Prediction Visualization
Visualize predictions against historical data.

**Endpoint:** `POST /viz/visualize/predictions/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)
- `file`: CSV file with historical data
- `predictions`: Array of prediction objects

**Request Example:**
```bash
curl -X POST "http://localhost:8000/viz/visualize/predictions/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv" \
  -F 'predictions=[
    {
      "date": "2025-01-23",
      "recommended_price": 25.5,
      "predicted_sales": 150.0
    }
  ]'
```

**Response:**
```json
{
  "prediction_analysis": {
    // Plotly figure data for historical vs predicted
  },
  "exploration_analysis": {
    // Plotly figure data for exploration analysis
  }
}
```

### 3. Model Performance
Get model performance visualizations.

**Endpoint:** `GET /viz/visualize/model/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)

**Request Example:**
```bash
curl "http://localhost:8000/viz/visualize/model/woolball"
```

**Response:**
```json
{
  "historical_analysis": {
    // Plotly figure data
  },
  "training_performance": {
    // Training metrics
  },
  "data_coverage": {
    "total_records": 100,
    "price_records": 95,
    "sales_records": 95,
    "conversion_records": {
      "organic": 90,
      "ad": 85
    }
  }
}
```

### 4. Visualization History
Get all saved visualizations for a product.

**Endpoint:** `GET /viz/visualize/history/{product_id}`

**Parameters:**
- `product_id`: Product identifier (in URL)

**Request Example:**
```bash
curl "http://localhost:8000/viz/visualize/history/woolball"
```

**Response:**
```json
{
  "historical_analysis": {
    // Latest historical analysis
  },
  "prediction_analysis": {
    // Latest prediction analysis
  },
  "model_performance": {
    // Latest model performance
  },
  "prediction_history": [
    "predictions_20250221_120000",
    "predictions_20250221_130000"
  ]
}
```

## Data Format

### CSV File Structure
The input CSV files should have the following columns:
```csv
Report Date,Product Price,Organic Conversion Percentage,Ad Conversion Percentage,Total Profit,Total Sales,Predicted Sales
2025-01-22,,0,0,0,0,124.85
2025-01-21,23.5,10.42,22.5,144.29,302.38,213.33
```

Required columns:
- `Report Date`: Date in YYYY-MM-DD format
- `Product Price`: Price point (can be empty for future dates)
- `Total Sales`: Actual sales (can be empty for future dates)

Optional columns:
- `Organic Conversion Percentage`: Organic conversion rate
- `Ad Conversion Percentage`: Ad conversion rate
- `Total Profit`: Profit amount
- `Predicted Sales`: Sales forecast for future dates

## Using the Visualizations

The visualization endpoints return Plotly figure data that can be rendered in the frontend using Plotly.js:

```javascript
// Example using React and Plotly.js
import Plot from 'react-plotly.js';

function ProductAnalysis({ productId }) {
  const [vizData, setVizData] = useState(null);

  useEffect(() => {
    // Fetch visualization data
    fetch(`/viz/visualize/model/${productId}`)
      .then(res => res.json())
      .then(data => setVizData(data));
  }, [productId]);

  if (!vizData) return <div>Loading...</div>;

  return (
    <div>
      <Plot
        data={vizData.historical_analysis.data}
        layout={vizData.historical_analysis.layout}
      />
      {/* Render other visualizations similarly */}
    </div>
  );
}
```

## Error Handling

All endpoints return standard HTTP status codes with detailed error messages:

### Success Codes
- **200 OK**: Request succeeded
- **201 Created**: Resource created successfully (new model trained)

### Client Error Codes
- **400 Bad Request**: Invalid parameters or request format
  ```json
  {
    "detail": "Invalid CSV format: missing required columns Report Date, Product Price"
  }
  ```

- **404 Not Found**: Resource not found
  ```json
  {
    "detail": "No trained model found for product woolball"
  }
  ```

- **409 Conflict**: Resource already exists or state conflict
  ```json
  {
    "detail": "Model already exists for woolball. Use force=true to retrain.",
    "model_path": "/path/to/model/checkpoint.pt",
    "metadata": {
      "training_date": "2025-02-21T12:00:00",
      "data_range": {
        "start": "2024-10-01",
        "end": "2025-01-22"
      }
    }
  }
  ```

### Server Error Codes
- **500 Internal Server Error**: Server-side processing error
  ```json
  {
    "detail": "Training failed: Error during model optimization - gradient explosion detected"
  }
  ```

### Common Error Scenarios

1. **Model Management**
   - Model not found when predicting prices (404)
   - Attempt to train model that already exists (409)
   - Invalid model configuration parameters (400)

2. **Data Validation**
   - Missing required CSV columns (400)
   - Invalid date format in Report Date (400)
   - Invalid numeric values in Price/Sales columns (400)
   - Future dates in historical data (400)

3. **Training Process**
   - Insufficient data points for training (400)
   - Gradient explosion/vanishing during training (500)
   - Memory allocation errors during training (500)
   - Wandb logging failures (500)

4. **Prediction Process**
   - Invalid future dates format (400)
   - Date range too far in future (400)
   - Model checkpoint corruption (500)
   - Numerical instability in predictions (500)

5. **Visualization**
   - Invalid data format for plotting (400)
   - Memory error during plot generation (500)
   - Too many data points for visualization (400)

### Error Prevention Tips
1. Always validate CSV format before upload
2. Check model existence before training
3. Use force=true parameter when retraining is intended
4. Ensure date formats follow YYYY-MM-DD
5. Keep prediction requests within reasonable future range (max 30 days)