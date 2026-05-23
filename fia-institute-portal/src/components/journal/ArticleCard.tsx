import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ArticleCardProps {
  title: string;
  authors: string;
  abstract?: string | null;
  doi_url?: string | null;
}

export function ArticleCard({ title, authors, abstract, doi_url }: ArticleCardProps) {
  return (
    <div className="border border-border rounded-xl p-6 bg-background">
      <h3 className="font-serif text-lg font-semibold text-foreground mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground mb-3">{authors}</p>
      {abstract && (
        <p className="text-sm text-foreground/80 leading-relaxed mb-4 line-clamp-4">{abstract}</p>
      )}
      {doi_url && (
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground font-mono break-all">{doi_url}</span>
          <Button variant="outline" size="sm" asChild>
            <a href={doi_url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="w-4 h-4 mr-1" />
              View on Zenodo
            </a>
          </Button>
        </div>
      )}
    </div>
  );
}
