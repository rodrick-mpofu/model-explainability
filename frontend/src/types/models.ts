export interface Model {
  name: string;
  display_name: string;
}

export interface Technique {
  name: string;
  display_name: string;
}

export interface AnalysisRequest {
  modelName: string;
  technique: 'gradcam' | 'shap';
  confidenceThreshold: number;
  topN: number;
}

export interface AnalysisResult {
  label: string;
  confidence: number;
  gradcam_img_path?: string;
  gradcam_img_url?: string;
}

export interface AnalysisResponse {
  success: boolean;
  results: AnalysisResult[];
  metadata: {
    model_used: string;
    technique_used: string;
    confidence_threshold: number;
    total_predictions: number;
  };
}

export interface UploadedFile {
  file: File;
  preview: string;
}
