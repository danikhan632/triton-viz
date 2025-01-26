// GridViewComponent.jsx
import React, { useState, useEffect, useMemo, memo } from 'react';
import {
  Box,
  Slider,
  Typography,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
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

/** 
 * Memoized grid component. 
 * Renders blocks based on xValues, yValues, zValues, and a blockLookup table.
 */
const MemoizedGrid = memo(function MemoizedGrid({
  xValues,
  yValues,
  zValues,
  blockLookup,
  handleBlockClick
}) {
  return (
    <Grid container spacing={1}>
      {zValues.map((z) =>
        yValues.map((y) => (
          <Grid container item xs={12} spacing={1} key={`row-${y}-${z}`}>
            {xValues.map((x) => {
              const key = `${x}:${y}:${z}`;
              const blockData = blockLookup[key] || null; // Lookup block data

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
                      onClick={() => handleBlockClick(blockData)}
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
      )}
    </Grid>
  );
});

const GridViewComponent = ({ setCurrBlock }) => {
  const [globalData, setGlobalData] = useState(null); // Store global data
  const [maxValues, setMaxValues] = useState([0, 0, 0]); // Max X, Y, Z for the grid
  const [sliderValues, setSliderValues] = useState([-1, -1, -1]); // Slider values for X, Y, Z
  const [isInfoPopupOpen, setIsInfoPopupOpen] = useState(false); // Info dialog visibility

  // 1. Fetch visualization data on mount
  useEffect(() => {
    const fetchData = async () => {
      const data = await fetchVisualizationData();
      if (data) {
        setGlobalData(data);
        determineMaxValues(data);
      }
    };
    fetchData();
  }, []);

  // 2. Determine the maximum X, Y, Z values from the visualization data
  const determineMaxValues = (visualizationData) => {
    if (!Array.isArray(visualizationData) || visualizationData.length === 0) {
      setMaxValues([0, 0, 0]);
      return;
    }

    // Initialize maxVals
    const maxVals = [0, 0, 0];
    visualizationData.forEach((item) => {
      const { block_indices } = item;
      // block_indices is something like [x, y, z]
      if (Array.isArray(block_indices)) {
        block_indices.forEach((val, idx) => {
          if (idx < maxVals.length) {
            maxVals[idx] = Math.max(maxVals[idx], val);
          }
        });
      }
    });
    setMaxValues(maxVals);
  };

  // 3. Memoized block lookup (key: "x:y:z" -> block data).
  //    We assume globalData is an array of objects, each having { block_indices, ... } 
  //    If you need to store more than just "operations", adapt accordingly.
  const blockLookup = useMemo(() => {
    if (!Array.isArray(globalData)) return {};

    const lookup = {};
    for (const item of globalData) {
      // For example, item might be { block_indices: [x,y,z], operations: [...] }
      const { block_indices, operations } = item;
      if (block_indices && block_indices.length === 3) {
        const [x, y, z] = block_indices;
        lookup[`${x}:${y}:${z}`] = operations || [];
      }
    }
    return lookup;
  }, [globalData]);

  // 4. Handle slider changes
  const handleSliderChange = (index, newValue) => {
    const newSliderValues = [...sliderValues];
    newSliderValues[index] = newValue;
    setSliderValues(newSliderValues);
  };

  // 5. Handle block click
  const handleBlockClick = (blockData) => {
    setCurrBlock(blockData);
  };

  // 6. Create the array of xValues, yValues, zValues based on slider or full range
  const [xValues, yValues, zValues] = useMemo(() => {
    const [xMax, yMax, zMax] = maxValues;
    const [xSlider, ySlider, zSlider] = sliderValues;

    const range = (start, end) => {
      return Array.from({ length: end - start + 1 }, (_, i) => i + start);
    };

    const xs = xSlider === -1 ? range(0, xMax) : [xSlider];
    const ys = ySlider === -1 ? range(0, yMax) : [ySlider];
    const zs = zSlider === -1 ? range(0, zMax) : [zSlider];

    return [xs, ys, zs];
  }, [maxValues, sliderValues]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Grid Display Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
 
        <MemoizedGrid
          xValues={xValues}
          yValues={yValues}
          zValues={zValues}
          blockLookup={blockLookup}
          handleBlockClick={handleBlockClick}
        />
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
              {sliderValues[index] === -1
                ? 'All values'
                : `Value: ${sliderValues[index]}`}
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
      <Dialog
        open={isInfoPopupOpen}
        onClose={() => setIsInfoPopupOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Kernel Source Code</DialogTitle>
        <DialogContent>
          <Typography
            component="pre"
            sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
          >
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
