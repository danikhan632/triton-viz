import React, { useState, useCallback, Suspense, lazy } from 'react';
import { Box, Typography, Button } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import isEqual from 'lodash/isEqual';

const GridViewComponent = lazy(() => import('./components/GridViewComponent'));
const BlockView = lazy(() => import('./components/BlockView'));
const CodeViewerComponent = lazy(() => import('./components/CodeViewerComponent'));

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const App = () => {
  const [currBlock, setCurrBlock] = useState(null); 
  const [currLine, setCurrLine] = useState(null);   
  const [codeLines, setCodeLines] = useState([]);   
  const [loadingCode, setLoadingCode] = useState(false);
  const [errorCode, setErrorCode] = useState(null);
  const [processedData, setProcessedData] = useState(null); 
  const [codeLoaded, setCodeLoaded] = useState(false); 

  const updateProcessedData = useCallback((newData) => {
    if (!isEqual(newData, processedData)) {
      setProcessedData(newData);
    }
  }, [processedData]);

  const handleLoadCode = async () => {
    setLoadingCode(true);
    setErrorCode(null);
    try {
      const response = await fetch('/get_src', { method: 'GET' });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const text = await response.text();
      setCodeLines(text.split('\n'));
      setCodeLoaded(true);
    } catch (error) {
      console.error('Error fetching source code:', error);
      setErrorCode('Failed to load source code.');
    } finally {
      setLoadingCode(false);
    }
  };

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
        {}
        <Box sx={{ bgcolor: 'grey.800', p: 2 }}>
          <Typography variant="h4" align="center">
            GPU Profiling Tool
          </Typography>
          {}
          {!codeLoaded && (
            <Box sx={{ textAlign: 'center', mt: 1 }}>
              <Button variant="contained" onClick={handleLoadCode} disabled={loadingCode}>
                {loadingCode ? 'Loading...' : 'Load Source'}
              </Button>
            </Box>
          )}
        </Box>

        {}
        <Box
          sx={{
            display: 'flex',
            flexGrow: 1,
            minHeight: 0,
          }}
        >
          {}
          <Box
            sx={{
              flex: '0 0 60%',
              p: 2,
              overflow: 'auto',
              height: '100%',
            }}
          >
            <Suspense fallback={<div style={{ color: 'white' }}>Loading grid...</div>}>
              {currBlock === null ? (
                <GridViewComponent setCurrBlock={setCurrBlock} />
              ) : (
                <BlockView
                  currBlock={currBlock}
                  setCurrBlock={setCurrBlock}
                  currLine={currLine}
                  codeLines={codeLines}
                  setProcessedData={updateProcessedData}
                  processedData={processedData}
                />
              )}
            </Suspense>
          </Box>

          {}
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
            <Suspense fallback={<div style={{ color: 'white' }}>Loading code viewer...</div>}>
              {codeLoaded && (
                <CodeViewerComponent
                  currBlock={currBlock}
                  currLine={currLine}
                  setCurrLine={setCurrLine}
                  codeLines={codeLines}
                  loadingCode={loadingCode}
                  errorCode={errorCode}
                  processedData={processedData}
                />
              )}
            </Suspense>
            {}
            {!codeLoaded && !errorCode && (
              <Box sx={{ p: 2 }}>
                <Typography variant="body1">
                  Source code not loaded. Click "Load Source" above.
                </Typography>
              </Box>
            )}
            {errorCode && (
              <Box sx={{ p: 2 }}>
                <Typography variant="body1" color="error">
                  {errorCode}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;