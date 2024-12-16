// GridBlock.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

const GridBlock = ({ data, onClick }) => {
  return (
    <Box
      onClick={() => onClick(data)} // Call onClick with block data
      sx={{
        p: 1,
        cursor: 'pointer',
        '&:hover': { bgcolor: 'action.hover' },
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        border: 1,
        borderColor: 'grey.700',
        borderRadius: 1,
        bgcolor: 'grey.800',
      }}
    >
      <Typography variant="caption">{`${data.x},${data.y},${data.z}`}</Typography>
      <Typography variant="body2">Operations: {data.operations.length}</Typography>
    </Box>
  );
};

export default GridBlock;
