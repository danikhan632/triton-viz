import React, { useEffect, useState, startTransition, Suspense } from 'react';
import { Box, Typography, Button, CircularProgress, ButtonGroup } from '@mui/material';
import PropTypes from 'prop-types';
import { fetchAndLogBlockData, TensorsVisualization } from './visualization-components';

const BlockView = ({
  currBlock,
  setCurrBlock,
  currLine,
  codeLines,
  setProcessedData,
  processedData,
}) => {
  // State management
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [variables, setVariables] = useState({});
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [cameraControls, setCameraControls] = useState(null);
  const [sliceMode, setSliceMode] = useState({});
  const [sliceIndices, setSliceIndices] = useState({});

  // Tensor slice controls
  const toggleSliceMode = (varName, dims) => {
    if (dims.length === 3) {
      setSliceMode(prev => ({
        ...prev,
        [varName]: !prev[varName]
      }));
      // Initialize slice index if not already set
      setSliceIndices(prev => ({
        ...prev,
        [varName]: prev[varName] ?? 0
      }));
    }
  };

  const updateSliceIndex = (varName, increment) => {
    setSliceIndices(prev => {
      const currentIndex = prev[varName] ?? 0;
      const maxIndex = variables[varName]?.dims[2] - 1 ?? 0;
      return {
        ...prev,
        [varName]: increment 
          ? Math.min(currentIndex + 1, maxIndex)
          : Math.max(currentIndex - 1, 0)
      };
    });
  };

  // Line number finder
  const findLineNumber = (sourceLine) => {
    return codeLines.findIndex(line => line.trim() === sourceLine.trim()) + 1;
  };

  const extractDims = (shapeArray) => {
    if (!Array.isArray(shapeArray)) return [-1, -1, -1];
    const len = shapeArray.length;
    if (len === 1) return [shapeArray[0], -1, -1];
    if (len === 2) return [shapeArray[0], shapeArray[1], -1];
    if (len >= 3) return shapeArray.slice(0, 3);
    return [-1, -1, -1];
  };
  
  const processVariable = (value) => {
    if (!value || typeof value !== 'object') {
      return {
        value: value,
        dims: [-1, -1, -1],
        highlighted_indices: [],
        tensor_ptr: null,
        solo_ptr: false,
        slice_shape: null
      };
    }
  
    // If the variable contains a data array
    if ('data' in value) {
      const usedDims = value.shape ? extractDims(value.shape) : (value.dims || [-1, -1, -1]);
      return {
        value: value.data,
        dims: usedDims,
        highlighted_indices: value.highlighted_indices || [],
        tensor_ptr: null,
        solo_ptr: false,
        slice_shape: null
      };
    }
  
    // If the variable contains a value array (like offs_k)
    if ('value' in value && Array.isArray(value.value)) {
      const usedDims = value.shape ? extractDims(value.shape) : [-1, -1, -1];
      return {
        value: value.value,
        dims: usedDims,
        highlighted_indices: value.highlighted_indices || [],
        tensor_ptr: null,
        solo_ptr: false,
        slice_shape: null
      };
    }
  
    if ('tensor_ptr' in value) {
      const usedDims = value.shape 
        ? extractDims(value.shape) 
        : (value.dim ? extractDims(value.dim) : [-1, -1, -1]);
  
      return {
        value: null,
        dims: usedDims,
        highlighted_indices: value.highlighted_indices || [],
        tensor_ptr: value.tensor_ptr || null,
        solo_ptr: !!value.solo_ptr,
        slice_shape: value.slice_shape || null
      };
    }
  
    // Default case
    return {
      value: value,
      dims: [-1, -1, -1],
      highlighted_indices: [],
      tensor_ptr: null,
      solo_ptr: false,
      slice_shape: null
    };
  };
  
  

  const updateVariables = (data) => {
    if (!data?.results) return;

    const newVariables = {};
    data.results.forEach(result => {
      try {
        const resultLine = findLineNumber(result.source_line);
        if (resultLine <= currLine && result.changed_vars) {
          Object.entries(result.changed_vars).forEach(([key, value]) => {
            
            newVariables[key] = processVariable(value);
          });
        }
      } catch (error) {
        console.error('Error processing result:', error, result);
      }
    });
    setVariables(newVariables);
  };

  // Data fetching
  useEffect(() => {
    if (!currBlock) {
      startTransition(() => {
        setVariables({});
        setProcessedData(null);
      });
      return;
    }

    let isMounted = true;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchAndLogBlockData(
          currBlock.x,
          currBlock.y,
          currBlock.z
        );
        if (isMounted) {
          startTransition(() => {
            setProcessedData(data);
            updateVariables(data);
          });
        }
      } catch (err) {
        if (isMounted) setError('Failed to fetch block data.');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchData();
    return () => { isMounted = false; };
  }, [currBlock, setProcessedData]);

  // Update variables when line changes
  useEffect(() => {
    if (processedData) {
      startTransition(() => updateVariables(processedData));
    }
  }, [currLine, processedData]);

  // Camera controls
  const handleFocusTensor = (index) => {
    if (!cameraControls) return;
    const spacing = 50;
    const numTensors = tensorVariables.length;
    const totalWidth = (numTensors - 1) * spacing;
    const position = [-totalWidth / 2 + index * spacing, -9, -50];
    cameraControls.focusOnPosition(position);
  };

  // Filter variables
  const filterTensorVariables = (vars) => {

    return Object.entries(vars).filter(([, variable]) => {
      const validDims = variable.dims.filter(dim => dim > 0);
      return validDims.length >= 1 && validDims.length <= 3;
    });
  };

  const filterNonTensorVariables = (vars) => {
    return Object.entries(vars).filter(([, variable]) => {
      const validDims = variable.dims.filter(dim => dim > 0);
      
      return validDims.length === 0 || validDims.length > 3;
    });
  };

  const tensorVariables = filterTensorVariables(variables);
  const nonTensorVariables = filterNonTensorVariables(variables);

  // Render tensor controls
  const renderTensorControls = (key, variable, index) => {
    const is3DTensor = variable.dims.length === 3;
    const isSliceModeActive = sliceMode[key];
    const currentSliceIndex = sliceIndices[key] ?? 0;
    const maxSliceIndex = variable.dims[2] - 1;

    return (
      <Box key={key} sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', my: 1 }}>
        <Button
          variant="outlined"
          size="small"
          onClick={() => handleFocusTensor(index)}
        >
          Focus {key}
        </Button>
        
        {is3DTensor && (
          <>
            <Button
              variant="outlined"
              size="small"
              onClick={() => toggleSliceMode(key, variable.dims)}
            >
              {isSliceModeActive ? '3D View' : 'Slice View'}
            </Button>
            
            {isSliceModeActive && (
              <ButtonGroup size="small" variant="outlined">
                <Button
                  onClick={() => updateSliceIndex(key, false)}
                  disabled={currentSliceIndex === 0}
                >
                  Previous
                </Button>
                <Button disabled>
                  {currentSliceIndex + 1}/{maxSliceIndex + 1}
                </Button>
                <Button
                  onClick={() => updateSliceIndex(key, true)}
                  disabled={currentSliceIndex === maxSliceIndex}
                >
                  Next
                </Button>
              </ButtonGroup>
            )}
          </>
        )}
      </Box>
    );
  };

  return (
    <Box>
      <Button 
        onClick={() => setCurrBlock(null)} 
        variant="contained" 
        sx={{ mb: 2 }}
      >
        Back
      </Button>

      <Typography variant="h6">
        Block View for Block {currBlock ? `${currBlock.x},${currBlock.y},${currBlock.z}` : ''}
      </Typography>

      <Box sx={{ mt: 2, mb: 2, p: 2, border: '1px solid #ccc', borderRadius: 2 }}>
        <Typography variant="h6">Variables (Line {currLine}):</Typography>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {/* Non-tensor variables */}
            {nonTensorVariables.length > 0 && (
              <Box sx={{ mb: 2 }}>
                {nonTensorVariables.map(([key, variable]) => (
                  <Typography key={key} variant="body2">
                    {key}: {JSON.stringify(variable.value)}
                  </Typography>
                ))}
              </Box>
            )}

            {/* Tensor controls */}
            {tensorVariables.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => cameraControls?.resetView()}
                  sx={{ mb: 2 }}
                >
                  Reset View
                </Button>
                
                {tensorVariables.map(([key, variable], index) => 
                  renderTensorControls(key, variable, index)
                )}
              </Box>
            )}

            {/* Tensor visualization */}
            {tensorVariables.length > 0 ? (
              <Suspense fallback={<CircularProgress />}>
                <Box sx={{ height: '700px', width: '100%', mb: 2, position: 'relative' }}>
                  <TensorsVisualization 
                    tensorVariables={tensorVariables}
                    setHoveredInfo={setHoveredInfo}
                    onCameraControlsReady={setCameraControls}
                    sliceMode={sliceMode}
                    sliceIndices={sliceIndices}
                  />
                  
                  {/* Hover info overlay */}
                  {hoveredInfo && (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: '10px',
                        right: '10px',
                        padding: '10px',
                        backgroundColor: 'rgba(0,0,0,0.9)',
                        color: 'white',
                        borderRadius: '8px',
                        boxShadow: '0 0 10px rgba(0,0,0,0.3)',
                        zIndex: 1000,
                      }}
                    >
                      <Typography variant="body2">
                        <strong>{hoveredInfo.varName}</strong> [{hoveredInfo.indices.join(',')}]: {hoveredInfo.value}
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Suspense>
            ) : (
              <Typography variant="body2">
                No tensor variables to display.
              </Typography>
            )}

            {tensorVariables.length === 0 && nonTensorVariables.length === 0 && (
              <Typography variant="body2">No variables defined yet.</Typography>
            )}
          </>
        )}
      </Box>

      {error && (
        <Typography variant="body2" color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
    </Box>
  );
};

BlockView.propTypes = {
  currBlock: PropTypes.object,
  setCurrBlock: PropTypes.func.isRequired,
  currLine: PropTypes.number,
  codeLines: PropTypes.array.isRequired,
  setProcessedData: PropTypes.func.isRequired,
  processedData: PropTypes.object,
};

export default BlockView;