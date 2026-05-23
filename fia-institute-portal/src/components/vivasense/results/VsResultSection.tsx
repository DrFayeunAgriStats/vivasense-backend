/**
 * VsResultSection — shared, scholarly result section wrapper.
 *
 * Used by every VivaSense results surface (ANOVA, Genetics, Trait
 * Relationships) so that on-screen sections share the same hierarchy,
 * spacing and typography as the Word report headings.
 *
 * Supports BOTH:
 *  - Controlled mode (pass `isOpen` + `onToggle`)
 *  - Uncontrolled mode (pass `defaultOpen`)
 */

import * as React from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { VS_TYPOGRAPHY } from "@/lib/vivasenseDesignSystem";
import { cn } from "@/lib/utils";

interface VsResultSectionProps {
  title: string;
  icon?: React.ElementType;
  /** Optional small uppercase label above the title (e.g. "Section 3 · Results"). */
  eyebrow?: string;
  /** Optional one-line description under the title. */
  description?: string;
  /** Controlled open state. If provided, also pass onToggle. */
  isOpen?: boolean;
  onToggle?: () => void;
  /** Uncontrolled initial open state. */
  defaultOpen?: boolean;
  /** Right-aligned actions in the header (do not collapse on click). */
  actions?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

export function VsResultSection({
  title,
  icon: Icon,
  eyebrow,
  description,
  isOpen,
  onToggle,
  defaultOpen = true,
  actions,
  className,
  children,
}: VsResultSectionProps) {
  const isControlled = typeof isOpen === "boolean";
  const [internalOpen, setInternalOpen] = React.useState<boolean>(defaultOpen);
  const open = isControlled ? (isOpen as boolean) : internalOpen;

  const toggle = () => {
    if (isControlled) onToggle?.();
    else setInternalOpen((v) => !v);
  };

  return (
    <Card className={cn("overflow-hidden", className)}>
      <div
        className="cursor-pointer transition-colors hover:bg-muted/40 focus-visible:bg-muted/40 focus:outline-none"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
          }
        }}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            {/* Title block with optional eyebrow + accent rule */}
            <div className="min-w-0 flex-1">
              {eyebrow && (
                <div className={cn(VS_TYPOGRAPHY.sectionEyebrow, "mb-1")}>{eyebrow}</div>
              )}
              <div className="flex items-center gap-3">
                {Icon && <Icon className="h-5 w-5 shrink-0 text-primary" />}
                <h3 className={VS_TYPOGRAPHY.sectionTitle}>{title}</h3>
              </div>
              {description && (
                <p className="mt-1 text-sm text-muted-foreground">{description}</p>
              )}
            </div>

            <div className="flex items-center gap-2">
              {actions && (
                <div onClick={(e) => e.stopPropagation()}>{actions}</div>
              )}
              {open ? (
                <ChevronUp className="h-5 w-5 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
              )}
            </div>
          </div>
          {/* Subtle accent rule that mirrors the Word section heading underline */}
          <div className="mt-3 h-px w-full bg-gradient-to-r from-primary/30 via-border to-transparent" />
        </CardHeader>
      </div>

      {open && <CardContent className="pt-0">{children}</CardContent>}
    </Card>
  );
}
