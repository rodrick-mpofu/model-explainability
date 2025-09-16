import React from 'react';
import { Model, Technique } from '../types/models';

interface ModelSelectorProps {
  models: Model[];
  techniques: Technique[];
  selectedModel: string;
  selectedTechnique: 'gradcam' | 'shap';
  onModelChange: (model: string) => void;
  onTechniqueChange: (technique: 'gradcam' | 'shap') => void;
  disabled?: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  techniques,
  selectedModel,
  selectedTechnique,
  onModelChange,
  onTechniqueChange,
  disabled = false
}) => {
  return (
    <div className="space-y-6">
      <div>
        <label htmlFor="technique" className="block text-sm font-medium text-gray-700 mb-2">
          Explanation Technique
        </label>
        <select
          id="technique"
          value={selectedTechnique}
          onChange={(e) => onTechniqueChange(e.target.value as 'gradcam' | 'shap')}
          disabled={disabled}
          className="input-field"
        >
          {techniques.map((technique) => (
            <option key={technique.name} value={technique.name}>
              {technique.display_name}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">
          {selectedTechnique === 'gradcam' 
            ? 'Grad-CAM highlights important regions with heatmaps'
            : 'SHAP provides pixel-level importance values'
          }
        </p>
      </div>

      <div>
        <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-2">
          Model
        </label>
        <select
          id="model"
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={disabled}
          className="input-field"
        >
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.display_name}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">
          Choose a pre-trained model for analysis
        </p>
      </div>
    </div>
  );
};

export default ModelSelector;
