import React from 'react';
import { BookOpen, Brain, Target, Upload } from 'lucide-react';

const HowItWorks: React.FC = () => {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <BookOpen className="mx-auto h-12 w-12 text-primary-600 mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-2">How It Works</h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          This application helps you understand how deep learning models make predictions 
          by providing visual explanations of their decision-making process.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="text-center">
          <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
            <Upload className="h-6 w-6 text-primary-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">1. Upload Image</h3>
          <p className="text-sm text-gray-600">
            Upload an image you want to analyze. The app supports common image formats.
          </p>
        </div>

        <div className="text-center">
          <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
            <Brain className="h-6 w-6 text-primary-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">2. Select Model</h3>
          <p className="text-sm text-gray-600">
            Choose from various pre-trained models like VGG16, ResNet50, or EfficientNet.
          </p>
        </div>

        <div className="text-center">
          <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
            <Target className="h-6 w-6 text-primary-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">3. Choose Technique</h3>
          <p className="text-sm text-gray-600">
            Select Grad-CAM for heatmaps or SHAP for pixel-level importance analysis.
          </p>
        </div>

        <div className="text-center">
          <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
            <Brain className="h-6 w-6 text-primary-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">4. View Results</h3>
          <p className="text-sm text-gray-600">
            See predictions with confidence scores and visual explanations.
          </p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Explanation Techniques</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Grad-CAM</h4>
            <p className="text-sm text-gray-600 mb-3">
              Gradient-weighted Class Activation Mapping highlights important regions in the image 
              that contributed to the model's prediction using gradient information.
            </p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Shows which parts of the image the model focused on</li>
              <li>• Generates heatmaps with warm colors for important regions</li>
              <li>• Works with any CNN architecture</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">SHAP</h4>
            <p className="text-sm text-gray-600 mb-3">
              SHapley Additive exPlanations provides pixel-level importance values using 
              game-theoretic principles to explain individual predictions.
            </p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Assigns importance to each pixel</li>
              <li>• Provides consistent and locally accurate explanations</li>
              <li>• Shows both positive and negative contributions</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-2">Tips for Best Results</h3>
        <ul className="text-sm text-blue-800 space-y-2">
          <li>• Use clear, high-quality images for better analysis</li>
          <li>• Adjust the confidence threshold to filter low-quality predictions</li>
          <li>• Try different models to see how they interpret the same image</li>
          <li>• Compare Grad-CAM and SHAP explanations for deeper insights</li>
        </ul>
      </div>
    </div>
  );
};

export default HowItWorks;
