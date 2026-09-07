import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#12213a",
        mist: "#f5f7fb",
        line: "#e4e9f1",
        cobalt: "#3157d5",
      },
      boxShadow: {
        card: "0 18px 45px rgba(24, 42, 75, 0.07)",
      },
    },
  },
  plugins: [],
};

export default config;
