import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, Download } from "lucide-react";
import { useRef } from "react";

interface Props {
  htmlTables: Record<string, string>;
}

/** Generate a Word-compatible HTML document and trigger download */
function downloadAsWord(tableName: string, htmlContent: string) {
  const doc = `
<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:w="urn:schemas-microsoft-com:office:word"
      xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8">
<title>${tableName}</title>
<style>
  body { font-family: 'Times New Roman', Times, serif; font-size: 12pt; margin: 1in; }
  table { border-collapse: collapse; width: 100%; margin: 12pt 0; }
  th, td { border: 1px solid #000; padding: 4pt 8pt; text-align: left; font-size: 11pt; }
  th { background-color: #f2f2f2; font-weight: bold; }
  caption { font-weight: bold; margin-bottom: 6pt; text-align: left; font-size: 11pt; }
  p { margin: 6pt 0; }
  .table-title { font-weight: bold; font-size: 12pt; margin: 12pt 0 6pt 0; }
</style>
</head>
<body>
${htmlContent}
</body>
</html>`.trim();

  const blob = new Blob([doc], {
    type: "application/msword",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const safeName = tableName.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_]/g, "");
  a.download = `VivaSense_${safeName}_${new Date().toISOString().slice(0, 10)}.doc`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Download ALL html_tables as a single Word document */
function downloadAllAsWord(htmlTables: Record<string, string>) {
  const allContent = Object.entries(htmlTables)
    .map(([name, html]) => {
      const title = name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
      return `<h3 class="table-title">${title}</h3>\n${html}`;
    })
    .join("\n<br/>\n");

  downloadAsWord("All_Tables", allContent);
}

function formatTitle(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function HtmlTablesSection({ htmlTables }: Props) {
  const entries = Object.entries(htmlTables).filter(([, html]) => html && html.trim().length > 0);
  if (entries.length === 0) return null;

  return (
    <div className="space-y-6">
      {/* Download All button */}
      <div className="flex items-center justify-between">
        <h3 className="font-serif text-xl font-bold text-foreground flex items-center gap-2">
          <FileText className="w-5 h-5 text-primary" />
          Publishable Tables
        </h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => downloadAllAsWord(htmlTables)}
        >
          <Download className="w-4 h-4 mr-2" />
          Download All as Word
        </Button>
      </div>

      {entries.map(([name, html]) => (
        <Card key={name}>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-3 text-base">
                <FileText className="w-5 h-5 text-primary" />
                {formatTitle(name)}
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => downloadAsWord(formatTitle(name), html)}
              >
                <Download className="w-4 h-4 mr-2" />
                Word
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div
              className="overflow-x-auto prose prose-sm max-w-none dark:prose-invert
                [&_table]:w-full [&_table]:border-collapse [&_table]:text-sm
                [&_th]:text-left [&_th]:px-3 [&_th]:py-2 [&_th]:font-semibold [&_th]:text-foreground [&_th]:bg-muted/50 [&_th]:border-b [&_th]:border-border
                [&_td]:px-3 [&_td]:py-2 [&_td]:text-muted-foreground [&_td]:border-b [&_td]:border-border/50
                [&_tr]:hover:bg-muted/30"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
