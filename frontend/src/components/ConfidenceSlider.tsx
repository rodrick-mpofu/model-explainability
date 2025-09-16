import React from 'react';

interface ConfidenceSliderProps {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

const ConfidenceSlider: React.FC<ConfidenceSliderProps> = ({
  value,
  onChange,
  disabled = false
}) => {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <label htmlFor="confidence" className="text-sm font-medium text-gray-700">
          Confidence Threshold
        </label>
        <span className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded">
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      
      <div className="relative">
        <input
          type="range"
          id="confidence"
          min="0"
          max="1"
          step="0.05"
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
          <span>75%</span>
          <span>100%</span>
        </div>
      </div>
      
      <p className="text-xs text-gray-500">
        Only show predictions with confidence above this threshold
      </p>
      
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .slider:disabled::-webkit-slider-thumb {
          background: #9ca3af;
          cursor: not-allowed;
        }
        
        .slider:disabled::-moz-range-thumb {
          background: #9ca3af;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default ConfidenceSlider;
