import { ReactNode } from "react";
import { Layout } from "@/components/layout/Layout";
import { JournalNav } from "./JournalNav";

interface JournalLayoutProps {
  children: ReactNode;
}

export function JournalLayout({ children }: JournalLayoutProps) {
  return (
    <Layout>
      <JournalNav />
      {children}
    </Layout>
  );
}
