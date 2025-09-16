import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X } from 'lucide-react';
import { UploadedFile } from '../types/models';

interface ImageUploadProps {
  onFileSelect: (file: UploadedFile | null) => void;
  selectedFile: UploadedFile | null;
  disabled?: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ 
  onFileSelect, 
  selectedFile, 
  disabled = false 
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const preview = URL.createObjectURL(file);
      onFileSelect({ file, preview });
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    },
    multiple: false,
    disabled
  });

  const removeFile = () => {
    if (selectedFile) {
      URL.revokeObjectURL(selectedFile.preview);
      onFileSelect(null);
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Upload Image</h3>
      
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive 
              ? 'border-primary-500 bg-primary-50' 
              : 'border-gray-300 hover:border-gray-400'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-lg font-medium text-gray-900 mb-2">
            {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
          </p>
          <p className="text-sm text-gray-500">
            or click to select a file
          </p>
          <p className="text-xs text-gray-400 mt-2">
            Supports PNG, JPG, JPEG, GIF, BMP, WEBP
          </p>
        </div>
      ) : (
        <div className="relative">
          <div className="relative rounded-lg overflow-hidden bg-gray-100">
            <img
              src={selectedFile.preview}
              alt="Uploaded image"
              className="w-full h-64 object-contain"
            />
            <button
              onClick={removeFile}
              disabled={disabled}
              className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors disabled:opacity-50"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            File: {selectedFile.file.name} ({(selectedFile.file.size / 1024 / 1024).toFixed(2)} MB)
          </p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
