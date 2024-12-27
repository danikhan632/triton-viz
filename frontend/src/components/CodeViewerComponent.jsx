// src/components/CodeViewerComponent.jsx
import React, { useEffect, useRef, useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import PropTypes from 'prop-types';

const CodeViewerComponent = ({
  currBlock,
  currLine,
  setCurrLine,
  codeLines,
  loadingCode,
  errorCode,
  processedData,
}) => {
  const codeContainerRef = useRef(null);
  const [currentHighlightIndex, setCurrentHighlightIndex] = useState(0);
  const [isInfoPopupOpen, setIsInfoPopupOpen] = useState(false);
  const [showIR, setShowIR] = useState(false); // New state for toggling IR vs Source

  // Find highlights
  const linesToHighlight = useMemo(() => {
    if (!processedData || !processedData.results || !codeLines) {
      return [];
    }

    const highlightSet = new Set();
    processedData.results.forEach((result) => {
      const lineNum = findLineNumber(codeLines, result.source_line);
      if (lineNum !== -1) {
        highlightSet.add(lineNum);
      }
    });

    const highlightLines = processedData.results
      .map((result) => {
        const lineNum = findLineNumber(codeLines, result.source_line);
        return lineNum;
      })
      .filter((lineNum) => lineNum !== -1 && highlightSet.has(lineNum));

    return highlightLines;
  }, [processedData, codeLines]);

  useEffect(() => {
    if (linesToHighlight.length > 0) {
      setCurrentHighlightIndex(0);
      setCurrLine(linesToHighlight[0]);
    } else if (codeLines.length > 0) {
      setCurrentHighlightIndex(0);
      setCurrLine(1);
    }
  }, [linesToHighlight, codeLines, setCurrLine]);

  useEffect(() => {
    if (currLine && codeContainerRef.current) {
      const lineElement = document.getElementById(`code-line-${currLine}`);
      if (lineElement) {
        lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currLine]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === '.') {
        event.preventDefault();
        handleNextLine();
      } else if (event.key === ',') {
        event.preventDefault();
        handlePrevLine();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [currentHighlightIndex, linesToHighlight]);

  const handlePrevLine = () => {
    if (currentHighlightIndex > 0) {
      const newIndex = currentHighlightIndex - 1;
      setCurrentHighlightIndex(newIndex);
      setCurrLine(linesToHighlight[newIndex]);
    }
  };

  const handleNextLine = () => {
    if (currentHighlightIndex < linesToHighlight.length - 1) {
      const newIndex = currentHighlightIndex + 1;
      setCurrentHighlightIndex(newIndex);
      setCurrLine(linesToHighlight[newIndex]);
    }
  };

  function findLineNumber(lines, sourceLine) {
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim() === sourceLine.trim()) {
        return i + 1;
      }
    }
    return -1;
  }

  // Determine if IR is available
  let irContent = '';
  if (processedData && processedData.results) {
    const resultForLine = processedData.results.find((r) => {
      const lineNum = findLineNumber(codeLines, r.source_line);
      return lineNum === currLine;
    });
    if (resultForLine && resultForLine.ir && resultForLine.ir.trim() !== '') {
      irContent = resultForLine.ir;
    }
  }

  // Handle errors or loading states
  if (errorCode) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          bgcolor: 'red.800',
          p: 2,
          borderRadius: 1,
        }}
      >
        <Typography variant="h6" color="white">
          Error: {errorCode}
        </Typography>
      </Box>
    );
  }

  if (loadingCode) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          bgcolor: 'grey.800',
          p: 2,
          borderRadius: 1,
        }}
      >
        <Typography variant="h6" color="grey.500">
          Loading...
        </Typography>
      </Box>
    );
  }

  if (!currBlock) {
    return (
      <Typography variant="h6" color="grey.500" align="center" sx={{ mt: 2 }}>
        Select a block to view code
      </Typography>
    );
  }

  const viewerTitle = irContent ? (showIR ? 'IR Viewer' : 'Source Code Viewer') : 'Source Code Viewer';

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        position: 'relative',
      }}
    >
      {/* Header Section */}
      <Box
        sx={{
          flex: '0 0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          mb: 2,
          position: 'relative'
        }}
      >
        <Typography variant="h6" gutterBottom>
          {viewerTitle}
        </Typography>
        {irContent && (
          <Button
            variant="outlined"
            size="small"
            sx={{ position: 'absolute', right: 10 }}
            onClick={() => setShowIR((prev) => !prev)}
          >
            {showIR ? 'Show Source' : 'Show IR'}
          </Button>
        )}
      </Box>

      {/* Content Section */}
      {showIR && irContent ? (
        // IR Content
        <Box
          sx={{
            flex: '1 1 auto',
            overflow: 'auto',
            bgcolor: '#1e1e1e',
            color: '#d4d4d4',
            p: 2,
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            fontSize: '14px',
            lineHeight: '1.5',
          }}
        >
          <Typography component="pre">{irContent}</Typography>
        </Box>
      ) : (
        // Source Code
        <Box
          ref={codeContainerRef}
          sx={{
            flex: '1 1 auto',
            overflow: 'auto',
            bgcolor: '#1e1e1e',
            color: '#d4d4d4',
            p: 2,
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            fontSize: '14px',
            lineHeight: '1.5',
          }}
        >
          {codeLines.map((line, index) => {
            const lineNumber = index + 1;
            const isCurrent = currLine === lineNumber;

            return (
              <div
                key={index}
                id={`code-line-${lineNumber}`}
                style={{
                  backgroundColor: isCurrent
                    ? 'rgba(255, 255, 0, 0.7)'
                    : 'transparent',
                  padding: '0 5px',
                  cursor: 'default',
                }}
              >
                <span style={{ color: '#888', userSelect: 'none' }}>
                  {lineNumber.toString().padStart(3, ' ')}:
                </span>{' '}
                {line}
              </div>
            );
          })}
        </Box>
      )}

      {/* Navigation Buttons (only when showing source and multiple highlights) */}
      {linesToHighlight.length > 1 && !showIR && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            mt: 2,
            gap: 2,
          }}
        >
          <Button
            onClick={handlePrevLine}
            disabled={currentHighlightIndex === 0}
            variant="contained"
          >
            Previous
          </Button>
          <Button
            onClick={handleNextLine}
            disabled={currentHighlightIndex === linesToHighlight.length - 1}
            variant="contained"
          >
            Next
          </Button>
        </Box>
      )}

      {/* Info Button */}
      <Button
        onClick={() => setIsInfoPopupOpen(true)}
        variant="outlined"
        sx={{ position: 'absolute', bottom: 10, right: 10 }}
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
            {codeLines.join('\n') || 'No kernel source code available'}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsInfoPopupOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

CodeViewerComponent.propTypes = {
  currBlock: PropTypes.object,
  currLine: PropTypes.number,
  setCurrLine: PropTypes.func.isRequired,
  codeLines: PropTypes.array.isRequired,
  loadingCode: PropTypes.bool.isRequired,
  errorCode: PropTypes.string,
  processedData: PropTypes.object,
};

export default CodeViewerComponent;
