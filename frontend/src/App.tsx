import React, { useState, useEffect } from 'react';
import { QueryClient, QueryClientProvider, useQuery, useMutation } from 'react-query';
import { AlertCircle, CheckCircle } from 'lucide-react';

import ImageUpload from './components/ImageUpload';
import ModelSelector from './components/ModelSelector';
import ConfidenceSlider from './components/ConfidenceSlider';
import ResultsDisplay from './components/ResultsDisplay';
import HowItWorks from './components/HowItWorks';
import { apiService } from './services/api';
import { UploadedFile, AnalysisRequest, AnalysisResponse, Model, Technique } from './types/models';

const queryClient = new QueryClient();

const MainApp: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('vgg16');
  const [selectedTechnique, setSelectedTechnique] = useState<'gradcam' | 'shap'>('gradcam');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);
  const [results, setResults] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'upload' | 'results' | 'how-it-works'>('upload');

  // Fetch available models and techniques
  const { data: models = [] } = useQuery<Model[]>('models', apiService.getModels);
  const { data: techniques = [] } = useQuery<Technique[]>('techniques', apiService.getTechniques);
  
  // Health check
  const { data: healthStatus } = useQuery('health', apiService.healthCheck, {
    onError: (error) => {
      console.error('API Health check failed:', error);
    }
  });

  // Analysis mutation
  const analysisMutation = useMutation(apiService.analyzeImage, {
    onSuccess: (data: AnalysisResponse) => {
      console.log('Analysis successful:', data);
      setResults(data.results);
      setActiveTab('results');
    },
    onError: (error: any) => {
      console.error('Analysis failed:', error);
      console.error('Error details:', {
        message: error.message,
        response: error.response,
        status: error.response?.status,
        data: error.response?.data
      });
    },
  });

  const handleAnalyze = () => {
    if (!selectedFile) {
      console.error('No file selected');
      return;
    }

    const request: AnalysisRequest = {
      modelName: selectedModel,
      technique: selectedTechnique,
      confidenceThreshold,
      topN: 5
    };

    console.log('Starting analysis with:', {
      file: selectedFile.file,
      fileName: selectedFile.file.name,
      fileSize: selectedFile.file.size,
      request
    });

    analysisMutation.mutate({ file: selectedFile.file, request });
  };

  const handleFileSelect = (file: UploadedFile | null) => {
    setSelectedFile(file);
    if (file) {
      setResults([]); // Clear previous results
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Model Explainability</h1>
              <p className="text-sm text-gray-600">TypeScript Version v2.0.0</p>
            </div>
            <div className="flex space-x-2">
              {healthStatus ? (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  <CheckCircle className="w-3 h-3 mr-1" />
                  API Connected ({healthStatus.mode})
                </span>
              ) : (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  Connecting...
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <nav className="flex space-x-8">
            {[
              { id: 'upload', label: 'Analyze Image' },
              { id: 'results', label: 'Results' },
              { id: 'how-it-works', label: 'How It Works' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        {activeTab === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Controls */}
            <div className="space-y-6">
              <ImageUpload
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                disabled={analysisMutation.isLoading}
              />

              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Settings</h3>
                <div className="space-y-6">
                  <ModelSelector
                    models={models}
                    techniques={techniques}
                    selectedModel={selectedModel}
                    selectedTechnique={selectedTechnique}
                    onModelChange={setSelectedModel}
                    onTechniqueChange={setSelectedTechnique}
                    disabled={analysisMutation.isLoading}
                  />

                  <ConfidenceSlider
                    value={confidenceThreshold}
                    onChange={setConfidenceThreshold}
                    disabled={analysisMutation.isLoading}
                  />

                  <button
                    onClick={handleAnalyze}
                    disabled={!selectedFile || analysisMutation.isLoading}
                    className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {analysisMutation.isLoading ? (
                      <div className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Analyzing...
                      </div>
                    ) : (
                      'Analyze Image'
                    )}
                  </button>
                </div>
              </div>

              {/* Error Display */}
              {analysisMutation.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex">
                    <AlertCircle className="h-5 w-5 text-red-400 mr-2 mt-0.5" />
                    <div>
                      <h3 className="text-sm font-medium text-red-800">Analysis Failed</h3>
                      <p className="text-sm text-red-700 mt-1">
                        {analysisMutation.error?.response?.data?.detail || 
                         analysisMutation.error?.message || 
                         'An unexpected error occurred'}
                      </p>
                      {process.env.NODE_ENV === 'development' && analysisMutation.error && (
                        <details className="mt-2 text-xs text-red-600">
                          <summary className="cursor-pointer">Debug Info</summary>
                          <pre className="mt-1 p-2 bg-red-100 rounded text-xs overflow-auto">
                            {JSON.stringify({
                              status: analysisMutation.error?.response?.status,
                              data: analysisMutation.error?.response?.data,
                              message: analysisMutation.error?.message
                            }, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Preview */}
            <div>
              {selectedFile && (
                <div className="card p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Image Preview</h3>
                  <div className="relative rounded-lg overflow-hidden bg-gray-100">
                    <img
                      src={selectedFile.preview}
                      alt="Selected image"
                      className="w-full h-64 object-contain"
                    />
                  </div>
                  <div className="mt-4 text-sm text-gray-600">
                    <p><strong>File:</strong> {selectedFile.file.name}</p>
                    <p><strong>Size:</strong> {(selectedFile.file.size / 1024 / 1024).toFixed(2)} MB</p>
                    <p><strong>Type:</strong> {selectedFile.file.type}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <ResultsDisplay
            results={results}
            technique={selectedTechnique}
            isLoading={analysisMutation.isLoading}
          />
        )}

        {activeTab === 'how-it-works' && <HowItWorks />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>Model Explainability v2.0.0 - TypeScript Implementation</p>
            <p className="mt-1">
              Built with React, TypeScript, FastAPI, and your existing Python AI modules
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <MainApp />
    </QueryClientProvider>
  );
};

export default App;
