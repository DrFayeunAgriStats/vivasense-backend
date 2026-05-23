import { ReactNode } from "react";
import { VivaSenseHeader } from "./VivaSenseHeader";
import { VivaSenseFooter } from "./VivaSenseFooter";

export function VivaSenseLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-[#0b1d14] text-white">
      <VivaSenseHeader />
      <main className="flex-1">{children}</main>
      <VivaSenseFooter />
    </div>
  );
}
