import { parseRoomformScene } from "./index";

export const demoRoom = parseRoomformScene({
  schema: "roomform.scene.v1",
  id: "demo-living-room",
  name: "Living room study",
  units: "meters",
  coordinateSystem: "right-handed-y-up",
  shell: {
    floor: [
      [-3.2, -2.5],
      [3.2, -2.5],
      [3.2, 2.5],
      [-3.2, 2.5],
    ],
    ceilingHeight: 2.7,
    walls: [
      {
        id: "wall-north",
        start: [-3.2, -2.5],
        end: [3.2, -2.5],
        height: 2.7,
        openings: [
          {
            id: "window-north",
            kind: "window",
            offset: 2.15,
            width: 2.1,
            height: 1.25,
            sillHeight: 0.85,
          },
        ],
      },
      {
        id: "wall-east",
        start: [3.2, -2.5],
        end: [3.2, 2.5],
        height: 2.7,
      },
      {
        id: "wall-south",
        start: [3.2, 2.5],
        end: [-3.2, 2.5],
        height: 2.7,
        openings: [
          {
            id: "door-south",
            kind: "door",
            offset: 0.55,
            width: 0.9,
            height: 2.08,
          },
        ],
      },
      {
        id: "wall-west",
        start: [-3.2, 2.5],
        end: [-3.2, -2.5],
        height: 2.7,
      },
    ],
  },
  objects: [
    {
      id: "sofa-1",
      label: "Sofa",
      category: "sofa",
      transform: { position: [0, 0.45, 1.75], rotation: [0, 0, 0, 1] },
      originalTransform: { position: [0, 0.45, 1.75], rotation: [0, 0, 0, 1] },
      bounds: [2.4, 0.9, 0.85],
      asset: {
        kind: "glb",
        uri: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/SheenWoodLeatherSofa/glTF-Binary/SheenWoodLeatherSofa.glb",
      },
    },
    {
      id: "coffee-table-1",
      label: "Coffee table",
      category: "table",
      transform: { position: [0, 0.2, 0.35], rotation: [0, 0, 0, 1] },
      originalTransform: { position: [0, 0.2, 0.35], rotation: [0, 0, 0, 1] },
      bounds: [1.25, 0.4, 0.7],
    },
    {
      id: "chair-1",
      label: "Lounge chair",
      category: "chair",
      transform: { position: [-2.2, 0.48, 0.45], rotation: [0, 0.38, 0, 0.92] },
      originalTransform: {
        position: [-2.2, 0.48, 0.45],
        rotation: [0, 0.38, 0, 0.92],
      },
      bounds: [0.85, 0.95, 0.9],
      asset: {
        kind: "glb",
        uri: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/SheenChair/glTF-Binary/SheenChair.glb",
      },
    },
    {
      id: "console-1",
      label: "Media console",
      category: "cabinet",
      transform: { position: [0, 0.38, -2.15], rotation: [0, 0, 0, 1] },
      originalTransform: { position: [0, 0.38, -2.15], rotation: [0, 0, 0, 1] },
      bounds: [1.9, 0.76, 0.4],
    },
    {
      id: "lamp-1",
      label: "Floor lamp",
      category: "lamp",
      transform: { position: [2.5, 0.85, 1.75], rotation: [0, 0, 0, 1] },
      originalTransform: { position: [2.5, 0.85, 1.75], rotation: [0, 0, 0, 1] },
      bounds: [0.45, 1.7, 0.45],
      asset: {
        kind: "glb",
        uri: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/IridescenceLamp/glTF-Binary/IridescenceLamp.glb",
      },
    },
  ],
});
