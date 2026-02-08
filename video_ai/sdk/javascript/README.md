# @video-ai/sdk

Official JavaScript/TypeScript SDK for Video AI Platform.

## Installation

```bash
npm install @video-ai/sdk
# or
yarn add @video-ai/sdk
# or
pnpm add @video-ai/sdk
```

## Quick Start

### Basic Generation

```typescript
import { VideoAIClient } from '@video-ai/sdk';

const client = new VideoAIClient({
  apiUrl: 'http://localhost:8000',
  apiKey: 'your-api-key', // optional
});

// Generate a video
const job = await client.generate({
  prompt: 'A majestic eagle soaring over snow-capped mountains at golden hour',
  width: 1280,
  height: 720,
  durationSeconds: 6,
  qualityPreset: 'balanced',
});

console.log(`Job started: ${job.jobId}`);

// Wait for completion
const result = await client.waitForCompletion(job.jobId, {
  onProgress: (progress, status) => {
    console.log(`Progress: ${progress}% - ${status}`);
  },
});

// Download the video
const blob = await client.download(result.jobId, 'eagle_video.mp4');
```

### Quick Function

```typescript
import { generateVideo } from '@video-ai/sdk';

// One-liner video generation
const videoBlob = await generateVideo('A sunset over the ocean with gentle waves', {
  apiUrl: 'http://localhost:8000',
  onProgress: (progress) => console.log(`${progress}%`),
});
```

### WebSocket Real-time Updates

```typescript
const client = new VideoAIClient({ apiUrl: 'http://localhost:8000' });

// Connect WebSocket
await client.connectWebSocket();

// Listen for events
client.on('progress', (data) => {
  console.log(`Job ${data.jobId}: ${data.progress}%`);
});

client.on('completed', (data) => {
  console.log(`Job ${data.jobId} completed!`);
});

client.on('failed', (data) => {
  console.error(`Job ${data.jobId} failed: ${data.error}`);
});

// Generate and subscribe
const job = await client.generate({ prompt: 'Your prompt here' });
client.subscribeToJob(job.jobId);

// Later: disconnect
client.disconnectWebSocket();
```

### Image-to-Video

```typescript
// From URL or base64
const job = await client.generateImageToVideo({
  prompt: 'The flowers sway gently in the wind',
  image: 'data:image/png;base64,...',
  motionStrength: 0.5,
  cameraMotion: 'slow_zoom_in',
});

// From File (browser)
const fileInput = document.getElementById('image-input') as HTMLInputElement;
const file = fileInput.files[0];

const job = await client.generateImageToVideo({
  prompt: 'The scene comes to life',
  image: file,
});
```

### Job Management

```typescript
// Get job status
const job = await client.getJob('job-id');
console.log(job.status, job.progress);

// List jobs
const jobs = await client.listJobs({
  status: 'completed',
  limit: 10,
});

// Cancel a job
await client.cancelJob('job-id');
```

### System Information

```typescript
// Check system status
const status = await client.getStatus();
console.log(`GPU: ${status.gpuName}`);
console.log(`VRAM Free: ${status.gpuVramFreeGb}GB`);
console.log(`Queue Size: ${status.queueSize}`);

// Get available models
const models = await client.getModels();
models.forEach(m => {
  console.log(`${m.name}: ${m.parameters} params, ${m.vramRequiredGb}GB VRAM`);
});

// Health check
const isHealthy = await client.healthCheck();
```

## API Reference

### VideoAIClient

#### Constructor

```typescript
new VideoAIClient({
  apiUrl: string;      // Required: API server URL
  apiKey?: string;     // Optional: API key for authentication
  timeout?: number;    // Optional: Request timeout in ms (default: 300000)
})
```

#### Methods

| Method | Description |
|--------|-------------|
| `generate(request)` | Start video generation |
| `generateImageToVideo(request)` | Generate video from image |
| `getJob(jobId)` | Get job status |
| `cancelJob(jobId)` | Cancel a job |
| `listJobs(options)` | List jobs |
| `waitForCompletion(jobId, options)` | Wait for job to complete |
| `download(jobId, filename?)` | Download generated video |
| `getDownloadUrl(jobId)` | Get download URL |
| `connectWebSocket(clientId?)` | Connect WebSocket |
| `subscribeToJob(jobId)` | Subscribe to job updates |
| `disconnectWebSocket()` | Disconnect WebSocket |
| `on(event, callback)` | Add event listener |
| `off(event, callback)` | Remove event listener |
| `getStatus()` | Get system status |
| `getModels()` | Get available models |
| `healthCheck()` | Check API health |

### Types

```typescript
interface GenerationRequest {
  prompt: string;
  negativePrompt?: string;
  width?: number;           // default: 1280
  height?: number;          // default: 720
  fps?: number;             // default: 24
  durationSeconds?: number; // default: 6.0
  modelName?: string;
  numInferenceSteps?: number; // default: 30
  guidanceScale?: number;     // default: 7.5
  seed?: number;
  qualityPreset?: 'fast' | 'balanced' | 'quality' | 'ultra';
  stylePreset?: string;
  usePreview?: boolean;
}

interface GenerationJob {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentStep?: string;
  estimatedRemainingSeconds?: number;
  createdAt?: string;
  completedAt?: string;
  error?: string;
  outputUrl?: string;
}
```

## Browser Usage

The SDK works in both Node.js and browser environments.

### CDN

```html
<script type="module">
  import { VideoAIClient } from 'https://unpkg.com/@video-ai/sdk/dist/index.esm.js';
  
  const client = new VideoAIClient({ apiUrl: 'http://localhost:8000' });
  // Use client...
</script>
```

## License

MIT
