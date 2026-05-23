import { useEffect, useState } from "react";
import { Newspaper } from "lucide-react";
import { JournalLayout } from "@/components/journal/JournalLayout";
import { ArticleCard } from "@/components/journal/ArticleCard";
import { supabase } from "@/integrations/supabase/client";

interface Article {
  id: string;
  title: string;
  authors: string;
  abstract: string | null;
  doi_url: string | null;
}

export default function JournalCurrentIssue() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase
      .from("journal_articles")
      .select("id, title, authors, abstract, doi_url")
      .eq("is_current_issue", true)
      .order("published_at", { ascending: false })
      .then(({ data }) => {
        setArticles(data || []);
        setLoading(false);
      });
  }, []);

  return (
    <JournalLayout>
      <section className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-8">
            Current Issue
          </h1>

          {loading ? (
            <p className="text-muted-foreground">Loading...</p>
          ) : articles.length === 0 ? (
            <div className="bg-muted/50 rounded-xl p-12 text-center border border-border">
              <Newspaper className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground text-lg">No articles published yet.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {articles.map((article) => (
                <ArticleCard key={article.id} {...article} />
              ))}
            </div>
          )}
        </div>
      </section>
    </JournalLayout>
  );
}
