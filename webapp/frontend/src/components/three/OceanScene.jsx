import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

// ─── Floating Bubble ────────────────────────────────────────────────────────
function Bubble({ position, speed = 0.3, size = 0.06 }) {
  const ref = useRef();
  const offset = useMemo(() => Math.random() * Math.PI * 2, []);
  useFrame((state) => {
    const t = state.clock.elapsedTime * speed + offset;
    ref.current.position.y += 0.004 * speed;
    ref.current.position.x = position[0] + Math.sin(t) * 0.12;
    if (ref.current.position.y > 8) ref.current.position.y = -8;
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[size, 8, 8]} />
      <meshStandardMaterial
        color="#00d4ff"
        transparent
        opacity={0.18}
        roughness={0}
        metalness={0.1}
      />
    </mesh>
  );
}

// ─── 3D Fish — pushed far to the sides, not centre ──────────────────────────
function Fish3D({ position, scale = 1, color = "#00b4d8", speed = 0.4 }) {
  const ref = useRef();
  const offset = useMemo(() => Math.random() * Math.PI * 2, []);
  const dir = useMemo(() => (Math.random() > 0.5 ? 1 : -1), []);

  useFrame((state) => {
    const t = state.clock.elapsedTime * speed + offset;
    ref.current.position.x += 0.007 * dir;
    ref.current.position.y = position[1] + Math.sin(t * 1.5) * 0.25;
    ref.current.rotation.z = Math.sin(t * 2) * 0.04;
    ref.current.rotation.y = dir > 0 ? 0 : Math.PI;
    if (ref.current.position.x > 16)  ref.current.position.x = -16;
    if (ref.current.position.x < -16) ref.current.position.x = 16;
  });

  return (
    <group ref={ref} position={position} scale={scale}>
      {/* Body */}
      <mesh>
        <sphereGeometry args={[0.28, 12, 8]} />
        <meshStandardMaterial color={color} roughness={0.3} metalness={0.3} />
      </mesh>
      {/* Head */}
      <mesh position={[0.24, 0, 0]}>
        <sphereGeometry args={[0.2, 10, 8]} />
        <meshStandardMaterial color={color} roughness={0.3} metalness={0.3} />
      </mesh>
      {/* Tail */}
      <mesh position={[-0.34, 0, 0]} rotation={[0, 0, 0.3]}>
        <coneGeometry args={[0.2, 0.32, 4]} />
        <meshStandardMaterial color={color} roughness={0.4} />
      </mesh>
      {/* Dorsal fin */}
      <mesh position={[0, 0.25, 0]} rotation={[0, 0, 0.1]}>
        <coneGeometry args={[0.09, 0.24, 4]} />
        <meshStandardMaterial color={color} roughness={0.4} />
      </mesh>
      {/* Eye */}
      <mesh position={[0.32, 0.07, 0.13]}>
        <sphereGeometry args={[0.045, 6, 6]} />
        <meshStandardMaterial color="white" />
      </mesh>
      <mesh position={[0.345, 0.07, 0.13]}>
        <sphereGeometry args={[0.028, 6, 6]} />
        <meshStandardMaterial color="#001020" />
      </mesh>
    </group>
  );
}

// ─── Particle Field ──────────────────────────────────────────────────────────
function ParticleField({ count = 500 }) {
  const ref = useRef();
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3]     = (Math.random() - 0.5) * 28;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 18;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 8 - 6;
    }
    return arr;
  }, [count]);

  useFrame((_, delta) => {
    ref.current.rotation.y += delta * 0.01;
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#00d4ff"
        size={0.035}
        sizeAttenuation
        depthWrite={false}
        opacity={0.45}
      />
    </Points>
  );
}

// ─── Seaweed — pushed to very edges ──────────────────────────────────────────
function Seaweed({ x, segments = 6 }) {
  const groupRef = useRef();
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (groupRef.current) {
      groupRef.current.children.forEach((child, i) => {
        child.rotation.z = Math.sin(t * 0.8 + i * 0.4) * 0.18;
      });
    }
  });
  return (
    <group ref={groupRef} position={[x, -5, -3]}>
      {Array.from({ length: segments }).map((_, i) => (
        <mesh key={i} position={[Math.sin(i * 0.3) * 0.1, i * 0.38, 0]}>
          <capsuleGeometry args={[0.045, 0.32, 4, 6]} />
          <meshStandardMaterial
            color={i % 2 === 0 ? "#0d9e5c" : "#0ab852"}
            roughness={0.6}
            emissive={i % 2 === 0 ? "#074d2d" : "#056929"}
            emissiveIntensity={0.3}
          />
        </mesh>
      ))}
    </group>
  );
}

// ─── Ocean Floor ─────────────────────────────────────────────────────────────
function OceanFloor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -6, 0]} receiveShadow>
      <planeGeometry args={[50, 25, 1, 1]} />
      <meshStandardMaterial color="#a8d8ea" roughness={1} />
    </mesh>
  );
}

// ─── Subtle light rays from above ────────────────────────────────────────────
function LightRays() {
  const rays = useMemo(() =>
    Array.from({ length: 6 }, (_, i) => ({
      x: (i - 3) * 2.5,
      rotZ: (Math.random() - 0.5) * 0.25,
      opacity: 0.025 + Math.random() * 0.03,
    })), []
  );
  const refs = useRef([]);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    refs.current.forEach((r, i) => {
      if (r) r.material.opacity = rays[i].opacity + Math.sin(t * 0.4 + i) * 0.015;
    });
  });
  return (
    <>
      {rays.map((ray, i) => (
        <mesh
          key={i}
          ref={(el) => (refs.current[i] = el)}
          position={[ray.x, 3, -7]}
          rotation={[0, 0, ray.rotZ]}
        >
          <coneGeometry args={[0.4, 12, 4, 1, true]} />
          <meshBasicMaterial
            color="#00aaff"
            transparent
            opacity={ray.opacity}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ))}
    </>
  );
}

// ─── Fish config — all far from centre (|x| > 5) ─────────────────────────────
const FISH_CONFIG = [
  { position: [-9,  1.0, -2], color: "#00b4d8", scale: 1.1,  speed: 0.32 },
  { position: [-7, -1.5, -3], color: "#f77f00", scale: 0.85, speed: 0.44 },
  { position: [ 8,  0.5, -2], color: "#48cae4", scale: 0.95, speed: 0.28 },
  { position: [ 6, -2.0, -3], color: "#90e0ef", scale: 0.7,  speed: 0.48 },
  { position: [-11, 0.0, -4], color: "#f4a261", scale: 0.8,  speed: 0.36 },
  { position: [ 10, 1.5, -4], color: "#2ec4b6", scale: 0.75, speed: 0.40 },
  { position: [-6,  2.5, -3], color: "#a8dadc", scale: 0.65, speed: 0.52 },
  { position: [ 7, -0.5, -2], color: "#e9c46a", scale: 0.9,  speed: 0.30 },
];

// Only put seaweed at extreme edges
const SEAWEED_X = [-12, -9, -6, 6, 9, 12];

const BUBBLE_CONFIG = Array.from({ length: 16 }, (_, i) => ({
  position: [
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 10,
    (Math.random() - 0.5) * 4 - 3,
  ],
  speed: 0.18 + Math.random() * 0.35,
  size:  0.035 + Math.random() * 0.055,
}));

// ─── Main export ──────────────────────────────────────────────────────────────
export default function OceanScene() {
  return (
    <>
      {/* Lighting — bright for light theme */}
      <ambientLight intensity={1.4} color="#ffffff" />
      <directionalLight position={[0, 12, 6]} intensity={1.6} color="#b8e4f7" castShadow />
      <pointLight position={[-8, 4, 3]}  intensity={1.0} color="#29b8e0" />
      <pointLight position={[ 8, -3, 3]} intensity={0.8} color="#56ccf2" />
      <pointLight position={[ 0, 6, 2]}  intensity={0.6} color="#ffffff" />

      {/* Atmospheric fog — light blue */}
      <fog attach="fog" args={["#cce8f4", 18, 40]} />
      <color attach="background" args={["#cce8f4"]} />

      <ParticleField count={500} />
      <LightRays />
      <OceanFloor />

      {FISH_CONFIG.map((f, i) => (
        <Fish3D key={i} {...f} />
      ))}

      {BUBBLE_CONFIG.map((b, i) => (
        <Bubble key={i} {...b} />
      ))}

      {SEAWEED_X.map((x, i) => (
        <Seaweed key={i} x={x} segments={5 + (i % 3)} />
      ))}
    </>
  );
}
