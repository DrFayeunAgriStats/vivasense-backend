import { Link, useLocation } from "react-router-dom";

const journalLinks = [
  { name: "About", href: "/journal/about" },
  { name: "Aims & Scope", href: "/journal/aims-scope" },
  { name: "Editorial Board", href: "/journal/editorial-board" },
  { name: "Author Guidelines", href: "/journal/author-guidelines" },
  { name: "Submit Manuscript", href: "/journal/submit" },
  { name: "Current Issue", href: "/journal/current-issue" },
  { name: "Archive", href: "/journal/archive" },
];

export function JournalNav() {
  const location = useLocation();

  return (
    <nav className="bg-muted/50 border-b border-border">
      <div className="container-wide">
        <div className="flex items-center gap-1 overflow-x-auto py-2 scrollbar-hide">
          <Link
            to="/journal"
            className={`px-3 py-2 text-sm font-medium rounded-lg whitespace-nowrap transition-colors ${
              location.pathname === "/journal"
                ? "text-primary bg-background shadow-sm"
                : "text-muted-foreground hover:text-foreground hover:bg-background/50"
            }`}
          >
            Home
          </Link>
          {journalLinks.map((link) => (
            <Link
              key={link.name}
              to={link.href}
              className={`px-3 py-2 text-sm font-medium rounded-lg whitespace-nowrap transition-colors ${
                location.pathname === link.href
                  ? "text-primary bg-background shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-background/50"
              }`}
            >
              {link.name}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
