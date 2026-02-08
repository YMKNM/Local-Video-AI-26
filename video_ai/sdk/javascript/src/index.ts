/**
 * Video AI JavaScript/TypeScript SDK
 *
 * Enterprise-ready SDK for Video AI Platform
 *
 * @example
 * ```typescript
 * import { VideoAIClient } from '@video-ai/sdk';
 *
 * const client = new VideoAIClient({ apiUrl: 'http://localhost:8000' });
 *
 * // Generate video
 * const job = await client.generate({ prompt: 'A sunset over the ocean' });
 *
 * // Wait for completion
 * const result = await client.waitForCompletion(job.jobId);
 *
 * // Download
 * await client.download(result.jobId, 'output.mp4');
 * ```
 */

// =============================================================================
// Types
// =============================================================================

export interface VideoAIClientConfig {
  apiUrl: string;
  apiKey?: string;
  timeout?: number;
}

export interface GenerationRequest {
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  fps?: number;
  durationSeconds?: number;
  modelName?: string;
  numInferenceSteps?: number;
  guidanceScale?: number;
  seed?: number;
  qualityPreset?: 'fast' | 'balanced' | 'quality' | 'ultra';
  stylePreset?: string;
  usePreview?: boolean;
}

export interface ImageToVideoRequest {
  prompt: string;
  image: string | File | Blob;
  motionStrength?: number;
  cameraMotion?: string;
  fps?: number;
  durationSeconds?: number;
}

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface GenerationJob {
  jobId: string;
  status: JobStatus;
  progress: number;
  currentStep?: string;
  estimatedRemainingSeconds?: number;
  createdAt?: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  outputUrl?: string;
  previewUrl?: string;
  metadata?: Record<string, any>;
}

export interface SystemStatus {
  status: string;
  gpuName: string;
  gpuVramTotalGb: number;
  gpuVramFreeGb: number;
  queueSize: number;
  activeJobs: number;
  uptimeSeconds: number;
  version: string;
}

export interface ModelInfo {
  name: string;
  type: string;
  parameters: string;
  maxResolution: [number, number];
  maxFrames: number;
  vramRequiredGb: number;
  supportedModes: string[];
}

export interface ProgressCallback {
  (progress: number, status: string): void;
}

export interface WebSocketMessage {
  type: string;
  jobId?: string;
  progress?: number;
  status?: string;
  error?: string;
  [key: string]: any;
}

// =============================================================================
// Video AI Client
// =============================================================================

export class VideoAIClient {
  private apiUrl: string;
  private apiKey?: string;
  private timeout: number;
  private ws?: WebSocket;
  private eventHandlers: Map<string, Set<Function>> = new Map();

  constructor(config: VideoAIClientConfig) {
    this.apiUrl = config.apiUrl.replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 300000; // 5 minutes default
  }

  // ===========================================================================
  // HTTP Helpers
  // ===========================================================================

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async request<T>(
    method: string,
    endpoint: string,
    body?: any,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.apiUrl}${endpoint}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: this.getHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
        ...options,
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorBody}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ===========================================================================
  // Generation Methods
  // ===========================================================================

  /**
   * Generate a video from a text prompt
   */
  async generate(request: GenerationRequest): Promise<GenerationJob> {
    const payload = {
      prompt: request.prompt,
      negative_prompt: request.negativePrompt || '',
      width: request.width || 1280,
      height: request.height || 720,
      fps: request.fps || 24,
      duration_seconds: request.durationSeconds || 6.0,
      model_name: request.modelName,
      num_inference_steps: request.numInferenceSteps || 30,
      guidance_scale: request.guidanceScale || 7.5,
      seed: request.seed,
      quality_preset: request.qualityPreset || 'balanced',
      style_preset: request.stylePreset,
      use_preview: request.usePreview || false,
    };

    const response = await this.request<any>('POST', '/generate', payload);

    return {
      jobId: response.job_id,
      status: response.status,
      progress: 0,
      metadata: {
        estimatedTime: response.estimated_time_seconds,
        queuePosition: response.queue_position,
      },
    };
  }

  /**
   * Generate video from an image
   */
  async generateImageToVideo(request: ImageToVideoRequest): Promise<GenerationJob> {
    let imageBase64: string;

    if (typeof request.image === 'string') {
      imageBase64 = request.image;
    } else if (request.image instanceof Blob || request.image instanceof File) {
      imageBase64 = await this.blobToBase64(request.image);
    } else {
      throw new Error('Invalid image type');
    }

    const payload = {
      prompt: request.prompt,
      image: imageBase64,
      motion_strength: request.motionStrength || 0.5,
      camera_motion: request.cameraMotion,
      fps: request.fps || 24,
      duration_seconds: request.durationSeconds || 4.0,
    };

    const response = await this.request<any>('POST', '/generate/image-to-video', payload);

    return {
      jobId: response.job_id,
      status: response.status,
      progress: 0,
    };
  }

  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]); // Remove data URL prefix
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // ===========================================================================
  // Job Management
  // ===========================================================================

  /**
   * Get job status and details
   */
  async getJob(jobId: string): Promise<GenerationJob> {
    const response = await this.request<any>('GET', `/jobs/${jobId}`);

    return {
      jobId: response.job_id,
      status: response.status,
      progress: response.progress || 0,
      currentStep: response.current_step,
      estimatedRemainingSeconds: response.estimated_remaining_seconds,
      createdAt: response.created_at,
      startedAt: response.started_at,
      completedAt: response.completed_at,
      error: response.error,
      outputUrl: response.output_url,
      previewUrl: response.preview_url,
      metadata: response.metadata,
    };
  }

  /**
   * Cancel a job
   */
  async cancelJob(jobId: string): Promise<boolean> {
    try {
      await this.request<any>('DELETE', `/jobs/${jobId}`);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * List jobs
   */
  async listJobs(options?: {
    status?: JobStatus;
    limit?: number;
    offset?: number;
  }): Promise<GenerationJob[]> {
    const params = new URLSearchParams();
    if (options?.status) params.set('status', options.status);
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());

    const queryString = params.toString();
    const endpoint = queryString ? `/jobs?${queryString}` : '/jobs';

    const response = await this.request<any>('GET', endpoint);

    return (response.jobs || []).map((job: any) => ({
      jobId: job.job_id || job.id,
      status: job.status,
      progress: job.progress || 0,
      createdAt: job.created_at,
    }));
  }

  /**
   * Wait for a job to complete
   */
  async waitForCompletion(
    jobId: string,
    options?: {
      pollInterval?: number;
      timeout?: number;
      onProgress?: ProgressCallback;
    }
  ): Promise<GenerationJob> {
    const pollInterval = options?.pollInterval || 2000;
    const timeout = options?.timeout;
    const startTime = Date.now();

    while (true) {
      const job = await this.getJob(jobId);

      if (options?.onProgress) {
        options.onProgress(job.progress, `Status: ${job.status}`);
      }

      if (job.status === 'completed') {
        return job;
      }

      if (job.status === 'failed') {
        throw new Error(`Job failed: ${job.error}`);
      }

      if (job.status === 'cancelled') {
        throw new Error('Job was cancelled');
      }

      if (timeout && Date.now() - startTime > timeout) {
        throw new Error(`Job ${jobId} did not complete within ${timeout}ms`);
      }

      await this.sleep(pollInterval);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Download generated video
   */
  async download(jobId: string, filename?: string): Promise<Blob> {
    const response = await fetch(`${this.apiUrl}/jobs/${jobId}/output`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.status}`);
    }

    const blob = await response.blob();

    // If running in browser and filename provided, trigger download
    if (typeof window !== 'undefined' && filename) {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    return blob;
  }

  /**
   * Get download URL for a job
   */
  getDownloadUrl(jobId: string): string {
    return `${this.apiUrl}/jobs/${jobId}/output`;
  }

  // ===========================================================================
  // WebSocket
  // ===========================================================================

  /**
   * Connect to WebSocket for real-time updates
   */
  async connectWebSocket(clientId?: string): Promise<void> {
    const wsUrl = this.apiUrl.replace('http', 'ws');
    const id = clientId || `sdk_${Date.now()}`;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${wsUrl}/ws/${id}`);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnect', {});
      };
    });
  }

  /**
   * Subscribe to job updates
   */
  subscribeToJob(jobId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify({ type: 'subscribe', job_id: jobId }));
  }

  /**
   * Add event listener
   */
  on(event: string, callback: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(callback);
  }

  /**
   * Remove event listener
   */
  off(event: string, callback: Function): void {
    this.eventHandlers.get(event)?.delete(callback);
  }

  private emit(event: string, data: any): void {
    this.eventHandlers.get(event)?.forEach((cb) => cb(data));
  }

  private handleWebSocketMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'progress':
        this.emit('progress', {
          jobId: message.job_id,
          progress: message.progress,
          status: message.status,
        });
        break;
      case 'job_completed':
        this.emit('completed', { jobId: message.job_id });
        break;
      case 'job_failed':
        this.emit('failed', { jobId: message.job_id, error: message.error });
        break;
      case 'subscribed':
        this.emit('subscribed', { jobId: message.job_id });
        break;
      default:
        this.emit('message', message);
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = undefined;
    }
  }

  // ===========================================================================
  // System
  // ===========================================================================

  /**
   * Get system status
   */
  async getStatus(): Promise<SystemStatus> {
    const response = await this.request<any>('GET', '/status');

    return {
      status: response.status,
      gpuName: response.gpu_name,
      gpuVramTotalGb: response.gpu_vram_total_gb,
      gpuVramFreeGb: response.gpu_vram_free_gb,
      queueSize: response.queue_size,
      activeJobs: response.active_jobs,
      uptimeSeconds: response.uptime_seconds,
      version: response.version,
    };
  }

  /**
   * Get available models
   */
  async getModels(): Promise<ModelInfo[]> {
    const response = await this.request<any[]>('GET', '/models');

    return response.map((m) => ({
      name: m.name,
      type: m.type,
      parameters: m.parameters,
      maxResolution: m.max_resolution,
      maxFrames: m.max_frames,
      vramRequiredGb: m.vram_required_gb,
      supportedModes: m.supported_modes,
    }));
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.request<any>('GET', '/health');
      return true;
    } catch {
      return false;
    }
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Quick function to generate a video
 */
export async function generateVideo(
  prompt: string,
  options?: Partial<GenerationRequest> & {
    apiUrl?: string;
    apiKey?: string;
    onProgress?: ProgressCallback;
  }
): Promise<Blob> {
  const client = new VideoAIClient({
    apiUrl: options?.apiUrl || 'http://localhost:8000',
    apiKey: options?.apiKey,
  });

  const job = await client.generate({ prompt, ...options });

  const result = await client.waitForCompletion(job.jobId, {
    onProgress: options?.onProgress,
  });

  return client.download(result.jobId);
}

// =============================================================================
// React Hook (if React is available)
// =============================================================================

/**
 * React hook for Video AI (if React is available)
 */
export function useVideoAI(config: VideoAIClientConfig) {
  // This would be implemented with React hooks
  // Left as a placeholder for the SDK
  return {
    generate: async (request: GenerationRequest) => {
      const client = new VideoAIClient(config);
      return client.generate(request);
    },
    getJob: async (jobId: string) => {
      const client = new VideoAIClient(config);
      return client.getJob(jobId);
    },
  };
}

// =============================================================================
// Export Default
// =============================================================================

export default VideoAIClient;
