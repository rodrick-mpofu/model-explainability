import React from 'react';
import { AnalysisResult } from '../types/models';
import { Brain, Target } from 'lucide-react';

interface ResultsDisplayProps {
  results: AnalysisResult[];
  technique: 'gradcam' | 'shap';
  isLoading?: boolean;
  metadata?: {
    model_used: string;
    technique_used: string;
    confidence_threshold: number;
    total_predictions: number;
    mode?: string;
  };
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  results,
  technique,
  isLoading = false,
  metadata
}) => {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">Analysis Results</h3>
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          <span className="ml-3 text-gray-600">Analyzing image...</span>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">Analysis Results</h3>
        <div className="text-center py-12">
          <Target className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-gray-600">No predictions found above the confidence threshold.</p>
          <p className="text-sm text-gray-500 mt-2">Try lowering the confidence threshold or uploading a different image.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-900">Analysis Results</h3>
        </div>
        {metadata?.mode && (
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            metadata.mode === 'real_ai' 
              ? 'bg-green-100 text-green-800' 
              : metadata.mode === 'fallback_mock'
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-blue-100 text-blue-800'
          }`}>
            {metadata.mode === 'real_ai' ? '🤖 Real AI' : 
             metadata.mode === 'fallback_mock' ? '⚠️ Fallback' : '🎭 Mock'}
          </span>
        )}
      </div>
      
      {metadata && (
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Model:</span>
              <p className="font-medium">{metadata.model_used}</p>
            </div>
            <div>
              <span className="text-gray-600">Technique:</span>
              <p className="font-medium">{metadata.technique_used}</p>
            </div>
            <div>
              <span className="text-gray-600">Threshold:</span>
              <p className="font-medium">{(metadata.confidence_threshold * 100).toFixed(0)}%</p>
            </div>
            <div>
              <span className="text-gray-600">Predictions:</span>
              <p className="font-medium">{metadata.total_predictions}</p>
            </div>
          </div>
        </div>
      )}
      
      <div className="space-y-6">
        {results.map((result, index) => (
          <div key={index} className="card p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Prediction Info */}
              <div className="space-y-4">
                <div>
                  <h4 className="text-lg font-semibold text-gray-900 mb-2">
                    Prediction #{index + 1}
                  </h4>
                  <div className="space-y-2">
                    <div>
                      <span className="text-sm font-medium text-gray-600">Category:</span>
                      <p className="text-lg font-semibold text-gray-900 capitalize">
                        {result.label.replace(/_/g, ' ')}
                      </p>
                    </div>
                    <div>
                      <span className="text-sm font-medium text-gray-600">Confidence:</span>
                      <div className="flex items-center space-x-2 mt-1">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-gray-900">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Visualization */}
              <div className="space-y-2">
                <h5 className="text-sm font-medium text-gray-600">
                  {technique === 'gradcam' ? 'Grad-CAM Heatmap' : 'SHAP Visualization'}
                </h5>
                {(result.gradcam_img_url || result.gradcam_img_path) && (
                  <div className="relative rounded-lg overflow-hidden bg-gray-100">
                    <img
                      src={result.gradcam_img_url || result.gradcam_img_path}
                      alt={`${technique} visualization`}
                      className="w-full h-64 object-contain"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                      }}
                    />
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  {technique === 'gradcam' 
                    ? 'Brighter areas indicate regions that most influenced the prediction'
                    : 'Color intensity shows pixel importance for the prediction'
                  }
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsDisplay;
