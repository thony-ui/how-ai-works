"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { 
  BrainIcon, 
  MenuIcon, 
  HomeIcon, 
  ChevronLeftIcon,
  ChevronRightIcon,
  TrendingUpIcon,
  Grid3x3Icon,
  PieChartIcon,
  LayersIcon,
  EyeIcon,
  NetworkIcon,
  TypeIcon,
  RocketIcon,
  SparklesIcon,
  TreeDeciduousIcon,
  ScatterChartIcon
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navSections = {
  main: [
    { href: '/', label: 'Home', icon: HomeIcon },
  ],
  fundamentals: [
    { href: '/gradient-descent', label: 'Gradient Descent', icon: TrendingUpIcon },
    { href: '/backpropagation', label: 'Backpropagation', icon: BrainIcon },
    { href: '/convolution', label: 'Convolution', icon: Grid3x3Icon },
  ],
  mlBasics: [
    { href: '/decision-tree', label: 'Decision Trees', icon: TreeDeciduousIcon },
    { href: '/k-means', label: 'K-Means Clustering', icon: ScatterChartIcon },
  ],
  advanced: [
    { href: '/softmax', label: 'Softmax & Cross-Entropy', icon: PieChartIcon },
    { href: '/multiclass', label: 'Multi-Class Classification', icon: LayersIcon },
    { href: '/attention', label: 'Attention Mechanism', icon: EyeIcon },
    { href: '/transformer', label: 'Transformer Block', icon: NetworkIcon },
    { href: '/token-prediction', label: 'Next-Token Prediction', icon: TypeIcon },
    { href: '/gpt-decoder', label: 'GPT Decoder', icon: RocketIcon },
  ]
};

// NavLinks component extracted outside to avoid render issues
function NavLinks({ 
  mobile = false, 
  onItemClick,
  sidebarOpen,
  pathname
}: { 
  mobile?: boolean; 
  onItemClick?: () => void;
  sidebarOpen?: boolean;
  pathname: string;
}) {
  return (
    <div className="space-y-6">
      {/* Main */}
      <div>
        {navSections.main.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          
          return (
            <Link key={item.href} href={item.href} onClick={onItemClick}>
              <Button
                variant={isActive ? 'default' : 'ghost'}
                className={cn(
                  "w-full justify-start gap-3",
                  mobile ? "" : sidebarOpen ? "" : "justify-center px-2"
                )}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {(mobile || sidebarOpen) && <span>{item.label}</span>}
              </Button>
            </Link>
          );
        })}
      </div>

      <Separator />

      {/* Fundamentals */}
      <div>
        {(mobile || sidebarOpen) && (
          <h3 className="px-3 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Fundamentals
          </h3>
        )}
        <div className="space-y-1">
          {navSections.fundamentals.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            
            return (
              <Link key={item.href} href={item.href} onClick={onItemClick}>
                <Button
                  variant={isActive ? 'default' : 'ghost'}
                  className={cn(
                    "w-full justify-start gap-3",
                    mobile ? "" : sidebarOpen ? "" : "justify-center px-2"
                  )}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {(mobile || sidebarOpen) && <span>{item.label}</span>}
                </Button>
              </Link>
            );
          })}
        </div>
      </div>

      <Separator />

      {/* ML Basics */}
      <div>
        {(mobile || sidebarOpen) && (
          <h3 className="px-3 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            ML Basics
          </h3>
        )}
        <div className="space-y-1">
          {navSections.mlBasics.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            
            return (
              <Link key={item.href} href={item.href} onClick={onItemClick}>
                <Button
                  variant={isActive ? 'default' : 'ghost'}
                  className={cn(
                    "w-full justify-start gap-3",
                    mobile ? "" : sidebarOpen ? "" : "justify-center px-2"
                  )}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {(mobile || sidebarOpen) && <span>{item.label}</span>}
                </Button>
              </Link>
            );
          })}
        </div>
      </div>

      <Separator />

      {/* Advanced */}
      <div>
        {(mobile || sidebarOpen) && (
          <h3 className="px-3 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Advanced Topics
          </h3>
        )}
        <div className="space-y-1">
          {navSections.advanced.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            
            return (
              <Link key={item.href} href={item.href} onClick={onItemClick}>
                <Button
                  variant={isActive ? 'default' : 'ghost'}
                  className={cn(
                    "w-full justify-start gap-3",
                    mobile ? "" : sidebarOpen ? "" : "justify-center px-2"
                  )}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {(mobile || sidebarOpen) && <span>{item.label}</span>}
                </Button>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function Navigation({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Top Bar */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Mobile Menu */}
            <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
              <SheetTrigger asChild className="md:hidden">
                <Button variant="ghost" size="sm">
                  <MenuIcon className="w-5 h-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-72">
                <SheetHeader>
                  <SheetTitle className="flex items-center gap-2">
                    <SparklesIcon className="w-5 h-5 text-blue-600" />
                    AI From First Principles
                  </SheetTitle>
                </SheetHeader>
                <div className="mt-6">
                  <NavLinks 
                    mobile 
                    onItemClick={() => setMobileOpen(false)} 
                    pathname={pathname}
                  />
                </div>
              </SheetContent>
            </Sheet>

            <Link href="/" className="flex items-center gap-2 group cursor-pointer">
              <SparklesIcon className="w-6 h-6 text-blue-600 group-hover:text-blue-700 transition-colors" />
              <span className="font-bold text-lg">AI From First Principles</span>
            </Link>
          </div>

          {/* Desktop Sidebar Toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="hidden md:flex"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? (
              <>
                <ChevronLeftIcon className="w-4 h-4 mr-2" />
                Collapse
              </>
            ) : (
              <>
                <ChevronRightIcon className="w-4 h-4 mr-2" />
                Expand
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Desktop Sidebar + Content */}
      <div className="flex min-h-[calc(100vh-73px)]">
        {/* Desktop Sidebar */}
        <aside
          className={cn(
            "hidden md:block border-r bg-white transition-all duration-300 sticky top-18.25 h-[calc(100vh-73px)] overflow-y-auto",
            sidebarOpen ? "w-64" : "w-16"
          )}
        >
          <div className="p-4">
            <NavLinks 
              sidebarOpen={sidebarOpen} 
              pathname={pathname}
            />
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 bg-linear-to-b from-gray-50 to-white">
          {children}
        </main>
      </div>
    </>
  );
}
