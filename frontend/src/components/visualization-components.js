import React, { useEffect, useRef, startTransition } from 'react';
import { Box } from '@mui/material';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import PropTypes from 'prop-types';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

const variableColorMap = new Map();

const getVariableColor = (varName) => {
  if (variableColorMap.has(varName)) {
    return variableColorMap.get(varName);
  }

  const hue = Math.random();
  const saturation = 0.7 + Math.random() * 0.3; 
  const lightness = 0.4 + Math.random() * 0.2;  

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

const fetchAndLogBlockData = async (gridX, gridY, gridZ) => {

  try {
    const response = await fetch('/process_blocks', {
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

    const data = await response.json();  
    console.log('Received data:', data); 
    return data;
  } catch (error) {
    console.error('Error fetching block data:', error);
    throw error;
  }
};

const CustomCameraControls = ({ onCameraReady }) => {
  const { camera, gl } = useThree();
  const cameraRotation = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const initialPosition = useRef(camera.position.clone());

  useEffect(() => {
    if (onCameraReady) {
      onCameraReady({
        focusOnPosition: (position) => {

          cameraRotation.current.set(0, 0, 0, 'YXZ');
          camera.setRotationFromEuler(cameraRotation.current);

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

function checkHighlights(highlightedIndices, cellIndex) {
  if (!Array.isArray(highlightedIndices) || highlightedIndices.length === 0) {
    return false;
  }

  return highlightedIndices.some(([hx, hy, hz]) => {
    return hx === cellIndex[0] && hy === cellIndex[1] && hz === cellIndex[2];
  });
}

const TensorMesh = React.memo(({
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
  const varColor = getVariableColor(varName);
  const validDims = dims.filter(d => d > 0);
  const dimCount = validDims.length;

  let rows = 1, cols = 1, depths = 1;
  if (dimCount === 1) {
    cols = validDims[0];
  } else if (dimCount === 2) {
    [rows, cols] = validDims;
  } else if (dimCount === 3) {
    [rows, cols, depths] = validDims;
  }

  const hasData = value && Array.isArray(value);
  let tensorData = [];
  let minVal = 0, maxVal = 1;

  if (hasData) {
    tensorData = value.flat(Infinity).filter(v => v != null);
    if (tensorData.length > 0) {
      minVal = Math.min(...tensorData);
      maxVal = Math.max(...tensorData);
    } else {
      tensorData = new Array(rows * cols * depths).fill(0);
    }
  } else if (isTensorPtr) {

    tensorData = new Array(rows * cols * depths).fill(0);
  } else {

    return <group position={position}></group>;
  }

  const boxes = [];
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < depths; k++) {
        const idx = i * cols * depths + j * depths + k;
        const val = tensorData[idx] ?? 0;

        const den = (maxVal - minVal) || 1;
        const intensity = maxVal === minVal ? 0.5 : (val - minVal) / den;

        const cubeIndex = [i, j, k];
        const highlighted = checkHighlights(highlightedIndices, cubeIndex);

        const cubeColor = getColorForValue(varColor, intensity);

        boxes.push(
          <group
            key={`${varName}-${i}-${j}-${k}`}
            position={[j - cols / 2, -i + rows / 2, k - depths / 2]}
          >
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();

                e.object.parent.traverse((child) => {
                  if (child.isLineSegments) {
                    child.material.color.set('yellow');
                  }
                });
                startTransition(() => {
                  setHoveredInfo({
                    varName,
                    indices: cubeIndex,
                    value: val,
                  });
                });
              }}
              onPointerOut={(e) => {
                e.stopPropagation();

                e.object.parent.traverse((child) => {
                  if (child.isLineSegments) {
                    child.material.color.set(highlighted ? 'red' : 'black');
                  }
                });
                startTransition(() => {
                  setHoveredInfo(null);
                });
              }}
            >
              <boxGeometry args={[0.9, 0.9, 0.9]} />
              <meshStandardMaterial color={cubeColor} />
            </mesh>

            {}
            <lineSegments>
              <edgesGeometry args={[new THREE.BoxGeometry(0.9, 0.9, 0.9)]} />
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

  return (
    <group position={position}>
      <Text
        position={[0, rows / 2 + 2, 0]}
        fontSize={1}
        color={`rgb(${varColor.r},${varColor.g},${varColor.b})`}
        anchorX="center"
        anchorY="middle"
      >
        {varName} {sliceMode ? `(Slice ${sliceIndex + 1}/${depths})` : ''}
      </Text>
      {boxes}
    </group>
  );
});

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
              value={value}
              dims={validDims}
              varName={key}
              highlightedIndices={highlighted_indices}
              setHoveredInfo={setHoveredInfo}
              position={tensorPosition}
              sliceMode={sliceMode[key]}
              sliceIndex={sliceIndices[key] || 0}
              isTensorPtr={highlighted_indices.length > 0} 
            />
          );
        })}
      </group>
    </Canvas>
  );
});

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