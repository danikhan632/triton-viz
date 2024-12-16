// src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import GridViewComponent from './components/GridViewComponent';
import CodeViewerComponent from './components/CodeViewerComponent';
import BlockView from './components/BlockView';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import isEqual from 'lodash/isEqual'; // Import lodash's isEqual for deep comparison

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const App = () => {
  const [currBlock, setCurrBlock] = useState(null); // Current selected block
  const [currLine, setCurrLine] = useState(null);   // Current highlighted line
  const [codeLines, setCodeLines] = useState([]);   // Source code lines
  const [loadingCode, setLoadingCode] = useState(true);
  const [errorCode, setErrorCode] = useState(null);
  const [processedData, setProcessedData] = useState(null); // Store processed data

  // Function to set processedData only if it's different
  const updateProcessedData = useCallback((newData) => {
    if (!isEqual(newData, processedData)) {
      setProcessedData(newData);
    }
  }, [processedData]);

  // Fetch the source code when the app mounts
  useEffect(() => {
    const fetchCode = async () => {
      try {
        const response = await fetch('/get_src', {
          method: 'GET',
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const text = await response.text();
        setCodeLines(text.split('\n'));
        setLoadingCode(false);
      } catch (error) {
        console.error('Error fetching source code:', error);
        setErrorCode('Failed to load source code.');
        setLoadingCode(false);
      }
    };

    fetchCode();
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          bgcolor: 'grey.900',
          color: 'white',
        }}
      >
        {/* Header */}
        <Box sx={{ bgcolor: 'grey.800', p: 2 }}>
          <Typography variant="h4" align="center">
            GPU Profiling Tool
          </Typography>
        </Box>

        {/* Main Content */}
        <Box
          sx={{
            display: 'flex',
            flexGrow: 1,
            minHeight: 0,
          }}
        >
          {/* Left Side: GridView or BlockView */}
          <Box
            sx={{
              flex: '0 0 60%',
              p: 2,
              overflow: 'auto',
              height: '100%',
            }}
          >
            {currBlock === null ? (
              <GridViewComponent setCurrBlock={setCurrBlock} />
            ) : (
              <BlockView
                currBlock={currBlock}
                setCurrBlock={setCurrBlock}
                currLine={currLine}
                codeLines={codeLines}
                setProcessedData={updateProcessedData} // Use the callback
                processedData={processedData} // Passed as a prop
              />
            )}
          </Box>

          {/* Right Side: CodeViewer */}
          <Box
            sx={{
              flex: '0 0 40%',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'flex-start',
              bgcolor: 'grey.800',
              height: '100%',
              overflow: 'auto',
            }}
          >
            <CodeViewerComponent
              currBlock={currBlock}
              currLine={currLine}
              setCurrLine={setCurrLine}
              codeLines={codeLines}
              loadingCode={loadingCode}
              errorCode={errorCode}
              processedData={processedData} // Passed as a prop
            />
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;
