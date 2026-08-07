// Unity-style hierarchy: layers and scene objects as a tree with
// per-node visibility eyes, click-to-select, double-click-to-focus.
import type { SplatObjectProposal } from "@roomform/scene/selection";
import {
  Armchair,
  Bed,
  Box,
  Boxes,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Lamp,
  Landmark,
  Monitor,
  ScanLine,
  Sofa,
  Tv,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useState } from "react";
import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

const CATEGORY_ICONS: Record<string, LucideIcon> = {
  bed: Bed,
  sofa: Sofa,
  couch: Sofa,
  chair: Armchair,
  armchair: Armchair,
  stool: Armchair,
  lamp: Lamp,
  tv: Tv,
  monitor: Monitor,
};

function categoryIcon(category: string): LucideIcon {
  const key = Object.keys(CATEGORY_ICONS).find((candidate) =>
    category.toLowerCase().includes(candidate),
  );
  return key ? CATEGORY_ICONS[key] : Box;
}

function EyeButton({
  visible,
  onToggle,
  disabled = false,
}: {
  visible: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  const Icon = visible ? Eye : EyeOff;
  return (
    <button
      type="button"
      aria-label={visible ? "Hide" : "Show"}
      aria-pressed={visible}
      disabled={disabled}
      className={cn(
        "flex size-5 shrink-0 items-center justify-center rounded-sm text-muted-foreground transition-opacity hover:text-foreground disabled:pointer-events-none disabled:opacity-30",
        visible ? "opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100" : "opacity-100 text-muted-foreground/60",
      )}
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
    >
      <Icon className="size-3.5" />
    </button>
  );
}

function TreeRow({
  depth,
  icon: Icon,
  label,
  dimmed = false,
  selected = false,
  trailing,
  onClick,
  onDoubleClick,
  chevron,
}: {
  depth: number;
  icon: LucideIcon;
  label: string;
  dimmed?: boolean;
  selected?: boolean;
  trailing?: ReactNode;
  onClick?: () => void;
  onDoubleClick?: () => void;
  chevron?: ReactNode;
}) {
  return (
    <div
      role="treeitem"
      aria-selected={selected}
      className={cn(
        "group/row flex h-7 cursor-default select-none items-center gap-1.5 rounded-sm pr-1 text-xs",
        selected ? "bg-accent text-accent-foreground" : "hover:bg-accent/50",
        dimmed && !selected && "text-muted-foreground/60",
      )}
      style={{ paddingLeft: `${depth * 14 + 6}px` }}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      {chevron ?? <span className="size-3.5 shrink-0" />}
      <Icon className="size-3.5 shrink-0 text-muted-foreground" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {trailing}
    </div>
  );
}

export type HierarchyLayer = {
  key: string;
  label: string;
  visible: boolean;
  onToggle: () => void;
};

export function HierarchyPanel({
  sceneTitle,
  layers,
  walls,
  objects,
  entitiesActive,
  hiddenIds,
  selectedId,
  onSelect,
  onToggleObject,
  onFocusObject,
}: {
  sceneTitle: string;
  layers: HierarchyLayer[];
  walls?: { count: number; visible: boolean; onToggle: () => void };
  objects: SplatObjectProposal[];
  /** whether object entities are currently rendered (view has entities on) */
  entitiesActive: boolean;
  hiddenIds: ReadonlySet<string>;
  selectedId?: string;
  onSelect: (id?: string) => void;
  onToggleObject: (id: string) => void;
  onFocusObject: (object: SplatObjectProposal) => void;
}) {
  const [objectsOpen, setObjectsOpen] = useState(true);
  const ObjectsChevron = objectsOpen ? ChevronDown : ChevronRight;

  return (
    <div className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b px-3">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Hierarchy
        </span>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div role="tree" className="p-1.5">
          <TreeRow depth={0} icon={Boxes} label={sceneTitle} onClick={() => onSelect(undefined)} />

          {layers.map((layer) => (
            <TreeRow
              key={layer.key}
              depth={1}
              icon={ScanLine}
              label={layer.label}
              dimmed={!layer.visible}
              trailing={<EyeButton visible={layer.visible} onToggle={layer.onToggle} />}
            />
          ))}

          {walls && (
            <TreeRow
              depth={1}
              icon={Landmark}
              label="Walls"
              dimmed={!walls.visible}
              trailing={
                <span className="flex items-center gap-1">
                  <Badge variant="outline" className="px-1 font-mono text-xs">
                    {walls.count}
                  </Badge>
                  <EyeButton visible={walls.visible} onToggle={walls.onToggle} />
                </span>
              }
            />
          )}

          {objects.length > 0 && (
            <>
              <TreeRow
                depth={1}
                icon={Box}
                label="Objects"
                dimmed={!entitiesActive}
                chevron={
                  <button
                    type="button"
                    aria-label={objectsOpen ? "Collapse" : "Expand"}
                    className="flex size-3.5 shrink-0 items-center justify-center text-muted-foreground"
                    onClick={(event) => {
                      event.stopPropagation();
                      setObjectsOpen((open) => !open);
                    }}
                  >
                    <ObjectsChevron className="size-3.5" />
                  </button>
                }
                trailing={
                  <Badge variant="outline" className="px-1 font-mono text-xs">
                    {objects.length}
                  </Badge>
                }
                onClick={() => setObjectsOpen((open) => !open)}
              />
              {objectsOpen &&
                objects.map((object) => {
                  const hidden = hiddenIds.has(object.id);
                  return (
                    <TreeRow
                      key={object.id}
                      depth={2}
                      icon={categoryIcon(object.category)}
                      label={object.label}
                      dimmed={hidden || !entitiesActive}
                      selected={object.id === selectedId}
                      trailing={
                        <EyeButton
                          visible={!hidden}
                          disabled={!entitiesActive}
                          onToggle={() => onToggleObject(object.id)}
                        />
                      }
                      onClick={() => onSelect(object.id === selectedId ? undefined : object.id)}
                      onDoubleClick={() => onFocusObject(object)}
                    />
                  );
                })}
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
