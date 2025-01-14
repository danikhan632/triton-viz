import React, { useEffect, useRef, startTransition, useMemo } from 'react';

import { Box } from '@mui/material';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import PropTypes from 'prop-types';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

// Store colors for variables
const variableColorMap = new Map();

// Generate a random color for a variable
const getVariableColor = (varName) => {
  if (variableColorMap.has(varName)) {
    return variableColorMap.get(varName);
  }

  // Generate a new color using HSL for better distribution
  const hue = Math.random();
  const saturation = 0.7 + Math.random() * 0.3; // 0.7-1.0
  const lightness = 0.4 + Math.random() * 0.2;  // 0.4-0.6

  // Convert HSL to RGB
  const h = hue;
  const s = saturation;
  const l = lightness;
  
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  
  const color = {
    r: Math.round(hue2rgb(p, q, h + 1/3) * 255),
    g: Math.round(hue2rgb(p, q, h) * 255),
    b: Math.round(hue2rgb(p, q, h - 1/3) * 255)
  };

  variableColorMap.set(varName, color);
  return color;
};

// Utility function to fetch block data
const fetchAndLogBlockData = async (gridX, gridY, gridZ) => {
  // console.log('Fetching data for block:', gridX, gridY, gridZ);
  try {
    const response = await fetch('http://10.0.0.77:5002/process_blocks', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        x: gridX,
        y: gridY,
        z: gridZ,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();  // Await the JSON response
    console.log('Received data:', data); // Log the data
    return data;
  } catch (error) {
    console.error('Error fetching block data:', error);
    throw error;
  }
};


// TensorMesh Component


const CustomCameraControls = ({ onCameraReady }) => {
  const { camera, gl } = useThree();
  const cameraRotation = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const initialPosition = useRef(camera.position.clone());

  useEffect(() => {
    if (onCameraReady) {
      onCameraReady({
        focusOnPosition: (position) => {
          // Reset rotation
          cameraRotation.current.set(0, 0, 0, 'YXZ');
          camera.setRotationFromEuler(cameraRotation.current);
          
          // Move camera to focus position
          camera.position.set(
            position[0],
            position[1] + 20,
            position[2] + 100
          );
          camera.lookAt(position[0], position[1], position[2]);
          camera.updateProjectionMatrix();
        },
        resetView: () => {
          camera.position.copy(initialPosition.current);
          cameraRotation.current.set(0, 0, 0, 'YXZ');
          camera.setRotationFromEuler(cameraRotation.current);
          camera.updateProjectionMatrix();
        }
      });
    }

    const canvas = gl.domElement;
    if (!canvas) return;

    const handleKeyDown = (event) => {
      const PAN_SPEED = 1;
      const ZOOM_SPEED = 2;
      const ROTATE_SPEED = 0.05;

      switch (event.key) {
        case 'ArrowUp':
          cameraRotation.current.x -= ROTATE_SPEED;
          cameraRotation.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, cameraRotation.current.x));
          camera.setRotationFromEuler(cameraRotation.current);
          break;
        case 'ArrowDown':
          cameraRotation.current.x += ROTATE_SPEED;
          cameraRotation.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, cameraRotation.current.x));
          camera.setRotationFromEuler(cameraRotation.current);
          break;
        case 'ArrowLeft':
          cameraRotation.current.y -= ROTATE_SPEED;
          camera.setRotationFromEuler(cameraRotation.current);
          break;
        case 'ArrowRight':
          cameraRotation.current.y += ROTATE_SPEED;
          camera.setRotationFromEuler(cameraRotation.current);
          break;
        case 'w':
        case 'W':
          camera.position.y += PAN_SPEED;
          break;
        case 's':
        case 'S':
          camera.position.y -= PAN_SPEED;
          break;
        case 'a':
        case 'A':
          camera.position.x -= PAN_SPEED;
          break;
        case 'd':
        case 'D':
          camera.position.x += PAN_SPEED;
          break;
        case 'o':
        case 'O':
          camera.position.z -= ZOOM_SPEED;
          break;
        case 'p':
        case 'P':
          camera.position.z += ZOOM_SPEED;
          break;
        default:
          break;
      }
      camera.updateProjectionMatrix();
    };

    const handleWheel = (event) => {
      const ZOOM_SPEED = 0.1;
      camera.position.z += event.deltaY * ZOOM_SPEED;
      camera.position.z = Math.max(20, Math.min(200, camera.position.z));
      camera.updateProjectionMatrix();
    };

    window.addEventListener('keydown', handleKeyDown);
    canvas.addEventListener('wheel', handleWheel);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [camera, gl, onCameraReady]);

  useFrame(() => {
    camera.updateProjectionMatrix();
  });

  return null;
};





// if (cubeIndex[1] === 47 && cubeIndex[0] === 31 ) {
//   console.log("triple: ",triple, " cubeIndex: ",cubeIndex);
//   return true;
// }

function getCornerValues(highlightedIndices) {
  if (!Array.isArray(highlightedIndices) || highlightedIndices.length === 0) {
    return null;
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < highlightedIndices.length; i++) {
    const row = highlightedIndices[i];
    if (Array.isArray(row)) {
      for (let j = 0; j < row.length; j++) {
        const triple = row[j];
        if (Array.isArray(triple) && triple.length === 3) {
          const [x, y, z] = triple;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (z < minZ) minZ = z;

          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
          if (z > maxZ) maxZ = z;
        }
      }
    }
  }

  return {
    minCorner: [minX, minY, minZ],
    maxCorner: [maxX, maxY, maxZ]
  };
}

function getFourCorners(highlightedIndices) {
  if (!Array.isArray(highlightedIndices) || highlightedIndices.length === 0) {
    return null;
  }

  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;

  for (let i = 0; i < highlightedIndices.length; i++) {
    const row = highlightedIndices[i];
    if (Array.isArray(row)) {
      for (let j = 0; j < row.length; j++) {
        const triple = row[j];
        if (Array.isArray(triple) && triple.length === 3) {
          const [x, y] = triple; // ignoring z for 2D corners
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }
  }

  // Return the four corners of the bounding rectangle
  return [
    [minX, minY],
     [maxX, minY],
   [minX, maxY],
    [maxX, maxY],
  ];
}



function checkHighlights(dimCount, highlightedIndices, cubeIndex, dims) {
  const foo =getFourCorners(highlightedIndices) ;
  if(foo !=null){
    console.log(foo);
  }
    
  const corners = getFourCorners(highlightedIndices);
  if (corners ===null) return false;
  for (let i = 0; i < corners.length; i++) {
    if (cubeIndex[1] === corners[i][0] && cubeIndex[0] === corners[i][1] ) {
      console.log("corners[i] ",corners[i], " cubeIndex: ",cubeIndex);
      return true;
    }

  }
  
  return false;
}


const TensorMesh = React.memo(
  ({
    value,
    dims,
    varName,
    highlightedIndices = [],
    setHoveredInfo,
    position,
    sliceMode = false,
    sliceIndex = 0,
    isTensorPtr
  }) => {
    // 1. All hooks at the top (no conditions!)
    const varColor = useMemo(() => getVariableColor(varName), [varName]);

    const validDims = useMemo(() => dims.filter((d) => d > 0), [dims]);
    const dimCount = validDims.length;

    const [rows, cols, depths] = useMemo(() => {
      if (dimCount === 1) return [1, validDims[0], 1];
      if (dimCount === 2) return [validDims[0], validDims[1], 1];
      if (dimCount === 3) return [validDims[0], validDims[1], validDims[2]];
      return [1, 1, 1];
    }, [validDims, dimCount]);

    // Flatten data + compute min/max
    const { tensorData, minVal, maxVal } = useMemo(() => {
      if (!value && !isTensorPtr) {
        // No data AND not a pointer → no tensor
        // We can still return something from here, but let's store data in a shape that
        // won't break the next hooks. We'll handle the "empty" scenario after all hooks.
        return {
          tensorData: [],
          minVal: 0,
          maxVal: 1
        };
      }

      let dataArray = [];
      let minV = 0;
      let maxV = 1;

      if (Array.isArray(value)) {
        dataArray = value.flat(Infinity).filter((v) => v != null);
        if (dataArray.length > 0) {
          minV = Math.min(...dataArray);
          maxV = Math.max(...dataArray);
        } else {
          dataArray = new Array(rows * cols * depths).fill(0);
        }
      } else if (isTensorPtr) {
        dataArray = new Array(rows * cols * depths).fill(0);
      }

      return {
        tensorData: dataArray,
        minVal: minV,
        maxVal: maxV
      };
    }, [value, rows, cols, depths, isTensorPtr]);

    // Reuse geometry
    const boxGeometry = useMemo(
      () => new THREE.BoxGeometry(0.9, 0.9, 0.9),
      []
    );
    const edgesGeometry = useMemo(
      () => new THREE.EdgesGeometry(boxGeometry),
      [boxGeometry]
    );

    // Create boxes
    const boxes = useMemo(() => {
      const result = [];
      const den = maxVal - minVal || 1;

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          for (let k = 0; k < depths; k++) {
            const idx = i * cols * depths + j * depths + k;
            const val = tensorData[idx] ?? 0;
            const intensity = maxVal === minVal ? 0.5 : (val - minVal) / den;
            const highlighted = checkHighlights(dimCount, highlightedIndices, [
              i,
              j,
              k
            ]);
            const cubeColor = getColorForValue(varColor, intensity);

            result.push(
              <group
                key={`${varName}-${i}-${j}-${k}`}
                position={[
                  j - cols / 2,
                  -i + rows / 2,
                  k - depths / 2
                ]}
              >
                <mesh
                  geometry={boxGeometry}
                  onPointerOver={(e) => {
                    e.stopPropagation();
                    e.object.parent.traverse((ch) => {
                      if (ch.isLineSegments) ch.material.color.set('yellow');
                    });
                    startTransition(() => {
                      setHoveredInfo({
                        varName,
                        indices: [i, j, k],
                        value: val
                      });
                    });
                  }}
                  onPointerOut={(e) => {
                    e.stopPropagation();
                    e.object.parent.traverse((ch) => {
                      if (ch.isLineSegments) {
                        ch.material.color.set(highlighted ? 'red' : 'black');
                      }
                    });
                    startTransition(() => {
                      setHoveredInfo(null);
                    });
                  }}
                >
                  <meshStandardMaterial color={cubeColor} />
                </mesh>
                <lineSegments geometry={edgesGeometry}>
                  <lineBasicMaterial
                    attach="material"
                    color={highlighted ? 'red' : 'black'}
                  />
                </lineSegments>
              </group>
            );
          }
        }
      }
      return result;
    }, [
      rows,
      cols,
      depths,
      tensorData,
      varName,
      varColor,
      minVal,
      maxVal,
      highlightedIndices,
      boxGeometry,
      edgesGeometry,
      setHoveredInfo,
      dimCount
    ]);

    // 2. Now we can do an "early return" AFTER we have used all Hooks
    // If there's really no data, just render an empty group
    const isDataEmpty = !tensorData || tensorData.length === 0;

    if (isDataEmpty && !isTensorPtr) {
      return <group position={position} />;
    }

    // 3. Otherwise, render the actual content
    return (
      <group position={position}>
        <Text
          position={[0, rows / 2 + 2, 0]}
          fontSize={1}
          color={`rgb(${varColor.r},${varColor.g},${varColor.b})`}
          anchorX="center"
          anchorY="middle"
        >
          {varName}
          {sliceMode ? ` (Slice ${sliceIndex + 1}/${depths})` : ''}
        </Text>
        {boxes}
      </group>
    );
  }
);



// Update TensorsVisualization to pass slice information
const TensorsVisualization = React.memo(({ 
  tensorVariables, 
  setHoveredInfo, 
  onCameraControlsReady,
  sliceMode,
  sliceIndices
}) => {
    const spacing = 50;
  const numTensors = tensorVariables.length;
  const totalWidth = (numTensors - 1) * spacing;

    return (
      <Canvas
        style={{ height: '100%', width: '100%' }}
        camera={{ position: [0, 0, 100], fov: 45 }}
      >
        <ambientLight />
        <pointLight position={[10, 10, 10]} />
        <CustomCameraControls onCameraReady={onCameraControlsReady} />
<group position={[-totalWidth / 2, 0, 0]}>
          {tensorVariables.map(([key, variable], index) => {
            const { value, dims, highlighted_indices = [], isTensorPtr } = variable;
            const validDims = dims.filter((dim) => dim > 0);
            const tensorPosition = [index * spacing, 0, 0];

            return (
              <TensorMesh
                key={key}
                value={variable.value}
                dims={validDims}
                varName={key}
                highlightedIndices={variable.highlighted_indices || []}
                setHoveredInfo={setHoveredInfo}
                position={tensorPosition}
                sliceMode={sliceMode[key]}
                sliceIndex={sliceIndices[key] || 0}
                isTensorPtr={variable.highlighted_indices.length > 0}
              />
            );
          })}
        </group>
      </Canvas>
    );
  }
);


// Helper function to get color based on intensity
const getColorForValue = (baseColor, intensity) => {
  const r = Math.round(255 - (255 - baseColor.r) * intensity);
  const g = Math.round(255 - (255 - baseColor.g) * intensity);
  const b = Math.round(255 - (255 - baseColor.b) * intensity);
  return `rgb(${r}, ${g}, ${b})`;
};


CustomCameraControls.propTypes = {
  onCameraReady: PropTypes.func,
};

TensorsVisualization.propTypes = {
  tensorVariables: PropTypes.arrayOf(PropTypes.array).isRequired,
  setHoveredInfo: PropTypes.func.isRequired,
  onCameraControlsReady: PropTypes.func,
};

export { fetchAndLogBlockData, TensorsVisualization };