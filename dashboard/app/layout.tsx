import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bid review | Anakin",
  description: "Review a grounded RFP response before submission.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
