// GridViewComponent.jsx
import React, { useState, useEffect } from 'react';
import { Box, Slider, Typography, Grid, Button, Dialog, DialogTitle, DialogContent, DialogActions } from '@mui/material';
import GridBlock from './GridBlock';

// Helper function to fetch data for visualization
const fetchVisualizationData = async () => {
  try {
    const response = await fetch('/api/data');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching visualization data:', error);
    return null;
  }
};

// GridViewComponent now handles only grid display and block selection
const GridViewComponent = ({ setCurrBlock }) => {
  const [globalData, setGlobalData] = useState(null);  // Store global data
  const [maxValues, setMaxValues] = useState([0, 0, 0]);  // Max X, Y, Z for the grid
  const [sliderValues, setSliderValues] = useState([-1, -1, -1]);  // Slider values for X, Y, Z axes
  const [isInfoPopupOpen, setIsInfoPopupOpen] = useState(false);  // Info dialog visibility

  useEffect(() => {
    // Fetch visualization data on component mount
    const fetchData = async () => {
      const data = await fetchVisualizationData();
      if (data && data.ops && data.ops.visualization_data) {
        setGlobalData(data);
        determineMaxValues(data.ops.visualization_data);
      }
    };
    fetchData();
  }, []);

  // Determine the maximum X, Y, Z values from the visualization data
  const determineMaxValues = (visualizationData) => {
    const keys = Object.keys(visualizationData);
    if (keys.length === 0) {
      setMaxValues([0, 0, 0]);
      return;
    }

    const maxVals = keys.reduce(
      (max, key) => {
        const [x, y, z] = key.split('_').map(Number);
        return [
          Math.max(max[0], x),
          Math.max(max[1], y),
          Math.max(max[2], z),
        ];
      },
      [0, 0, 0]
    );
    setMaxValues(maxVals);
  };

  // Update slider values for each axis
  const handleSliderChange = (index, newValue) => {
    const newSliderValues = [...sliderValues];
    newSliderValues[index] = newValue;
    setSliderValues(newSliderValues);
  };

  // Handle block click to set the selected block
  const handleBlockClick = (blockData) => {
    setCurrBlock(blockData);
  };

  // Render the grid based on slider and max values
  const renderGrid = () => {
    if (!globalData || !globalData.ops || !globalData.ops.visualization_data) return null;

    const [xMax, yMax, zMax] = maxValues;
    const [xSlider, ySlider, zSlider] = sliderValues;

    // Determine the range for each axis: if slider is -1, iterate over all values, otherwise use the slider value
    const xValues = xSlider === -1 ? Array.from({ length: xMax + 1 }, (_, i) => i) : [xSlider];
    const yValues = ySlider === -1 ? Array.from({ length: yMax + 1 }, (_, i) => i) : [ySlider];
    const zValues = zSlider === -1 ? Array.from({ length: zMax + 1 }, (_, i) => i) : [zSlider];

    return (
      <Grid container spacing={1}>
        {zValues.map((z) => (
          yValues.map((y) => (
            <Grid container item xs={12} spacing={1} key={`row-${y}-${z}`}>
              {xValues.map((x) => {
                const key = `${x}_${y}_${z}`;
                const blockData = globalData.ops.visualization_data[key];

                return (
                  <Grid item key={key}>
                    {blockData ? (
                      <GridBlock
                        data={{
                          x,
                          y,
                          z,
                          operations: blockData,
                        }}
                        onClick={handleBlockClick}
                      />
                    ) : (
                      <Box
                        sx={{
                          width: 50,
                          height: 50,
                          bgcolor: 'grey.800',
                          border: 1,
                          borderColor: 'grey.700',
                        }}
                      />
                    )}
                  </Grid>
                );
              })}
            </Grid>
          ))
        ))}
      </Grid>
    );
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Grid Display Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
        {renderGrid()}
      </Box>

      {/* Sliders for X, Y, Z axis */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
        {['X', 'Y', 'Z'].map((axis, index) => (
          <Box key={axis} sx={{ width: '30%' }}>
            <Typography>{axis} Axis</Typography>
            <Slider
              value={sliderValues[index]}
              onChange={(_, newValue) => handleSliderChange(index, newValue)}
              min={-1}
              max={maxValues[index]}
              step={1}
              marks
              valueLabelDisplay="auto"
            />
            <Typography variant="caption" sx={{ color: 'grey.500' }}>
              {sliderValues[index] === -1 ? 'All values' : `Value: ${sliderValues[index]}`}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* Info Button */}
      <Button
        onClick={() => setIsInfoPopupOpen(true)}
        variant="outlined"
        sx={{ position: 'absolute', top: 10, right: 10 }}
      >
        Info
      </Button>

      {/* Info Dialog */}
      <Dialog open={isInfoPopupOpen} onClose={() => setIsInfoPopupOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Kernel Source Code</DialogTitle>
        <DialogContent>
          <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {globalData?.kernel_src || 'No kernel source code available'}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsInfoPopupOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GridViewComponent;
