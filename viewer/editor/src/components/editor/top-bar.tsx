// Editor toolbar: project switching, save, evidence overlay, import.
import { FileUp, Keyboard, Save } from "lucide-react";
import type { ChangeEvent, RefObject } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export type SelectOption = { value: string; label: string };

function OptionSelect({
  value,
  options,
  onChange,
  width = "w-52",
}: {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  width?: string;
}) {
  return (
    <Select value={value} onValueChange={(next) => next && onChange(next)}>
      <SelectTrigger size="sm" className={`h-7 ${width} text-xs`}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value} className="text-xs">
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function TopBar({
  project,
  save,
  importInput,
  onImport,
}: {
  project?: { value: string; options: SelectOption[]; onChange: (id: string) => void };
  save?: { dirty: boolean; state: "idle" | "saving" | "error"; onSave: () => void };
  importInput: RefObject<HTMLInputElement | null>;
  onImport: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <header className="flex h-11 shrink-0 items-center gap-2 border-b bg-background px-3">
      <span className="text-sm font-semibold tracking-tight">Roomform</span>
      <Badge variant="outline" className="text-xs uppercase">
        editor
      </Badge>
      <Separator orientation="vertical" className="!h-5" />

      {project && (
        <OptionSelect
          value={project.value}
          options={project.options}
          onChange={project.onChange}
        />
      )}
      {save && (
        <>
          <Button
            variant={save.dirty ? "secondary" : "ghost"}
            size="sm"
            className="h-7 text-xs"
            disabled={save.state === "saving" || !save.dirty}
            onClick={save.onSave}
          >
            <Save className="size-3.5" />
            {save.state === "saving" ? "Saving…" : "Save"}
          </Button>
          <span
            className={`text-xs ${
              save.state === "error"
                ? "text-destructive"
                : "text-muted-foreground"
            }`}
          >
            {save.state === "error"
              ? "Save failed"
              : save.dirty
              ? "Unsaved changes"
              : "Saved"}
          </span>
        </>
      )}

      <div className="ml-auto flex items-center gap-1.5">
        <Tooltip>
          <TooltipTrigger
            render={
              <Button variant="ghost" size="icon" className="size-7">
                <Keyboard className="size-4" />
              </Button>
            }
          />
          <TooltipContent className="max-w-64">
            Left-drag orbit · Right-drag pan · Scroll zoom · ⌘-drag moves an object · WASD
            nudge · Q/E rotate · R reset · Esc deselect
          </TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={() => importInput.current?.click()}
              >
                <FileUp className="size-3.5" />
                Import
              </Button>
            }
          />
          <TooltipContent>Load an analysis JSON</TooltipContent>
        </Tooltip>
        <input
          ref={importInput}
          hidden
          type="file"
          accept="application/json,.json"
          onChange={onImport}
        />
      </div>
    </header>
  );
}
