import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Problem from "@/components/Problem";
import HowItWorks from "@/components/HowItWorks";
import ProductPreview from "@/components/ProductPreview";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-white antialiased">
      <Navbar />
      <Hero />
      <Problem />
      <HowItWorks />
      <ProductPreview />
      <CTA />
      <Footer />
    </main>
  );
}
