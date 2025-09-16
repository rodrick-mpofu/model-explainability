import axios from 'axios';
import { AnalysisRequest, AnalysisResponse, Model, Technique } from '../types/models';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for model processing
});

export const apiService = {
  // Get available models
  async getModels(): Promise<Model[]> {
    const response = await api.get('/models');
    return response.data.models;
  },

  // Get available techniques
  async getTechniques(): Promise<Technique[]> {
    const response = await api.get('/techniques');
    return response.data.techniques;
  },

  // Analyze image
  async analyzeImage({ file, request }: { file: File; request: AnalysisRequest }): Promise<AnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_name', request.modelName);
    formData.append('technique', request.technique);
    formData.append('confidence_threshold', request.confidenceThreshold.toString());
    formData.append('top_n', request.topN.toString());

    console.log('Sending analysis request:', {
      fileName: file.name,
      fileSize: file.size,
      modelName: request.modelName,
      technique: request.technique,
      confidenceThreshold: request.confidenceThreshold,
      topN: request.topN
    });

    try {
      const response = await api.post('/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Analysis response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Analysis error:', error);
      throw error;
    }
  },

  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },
};

export default apiService;
