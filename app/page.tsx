import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUpIcon, 
  BrainIcon, 
  ScanIcon,
  ArrowRightIcon,
  Grid3x3Icon
} from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: TrendingUpIcon,
      title: 'Gradient Descent',
      description: 'Visualize how gradient descent optimizes parameters in linear regression. Drag data points, adjust learning rates, and watch the algorithm converge.',
      href: '/gradient-descent',
      concepts: ['Loss Functions', 'Learning Rate', 'Convergence', 'Parameter Updates']
    },
    {
      icon: BrainIcon,
      title: 'Backpropagation',
      description: 'Step through forward and backward passes in a neural network. See how gradients flow backward to update weights using the chain rule.',
      href: '/backpropagation',
      concepts: ['Forward Pass', 'Backward Pass', 'Chain Rule', 'Weight Updates']
    },
    {
      icon: Grid3x3Icon,
      title: 'Convolution',
      description: 'Explore 2D convolution operations. Draw patterns, apply filters, and understand the foundation of convolutional neural networks.',
      href: '/convolution',
      concepts: ['Kernels', 'Feature Maps', 'Edge Detection', 'CNNs']
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <h1 className="text-6xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-blue-600 to-purple-600">
            AI From First Principles
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Turn core machine learning concepts into <span className="font-semibold text-foreground">step-by-step visual simulations</span>. 
            Interact with weights, gradients, and filters in real time.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto mt-8">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Card key={feature.title} className="group hover:shadow-xl transition-all duration-300 hover:scale-105 border-2">
                <CardHeader>
                  <div className="w-12 h-12 bg-linear-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                  <CardDescription className="text-sm leading-relaxed">
                    {feature.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-wrap gap-2">
                    {feature.concepts.map((concept) => (
                      <Badge key={concept} variant="secondary" className="text-xs">
                        {concept}
                      </Badge>
                    ))}
                  </div>
                  <Link href={feature.href}>
                    <Button className="w-full group-hover:bg-blue-600 transition-colors gap-2">
                      Explore
                      <ArrowRightIcon className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Advanced Topics */}
      <section className="container mx-auto px-6 py-16 pb-24">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-3">Advanced Topics</h2>
            <p className="text-muted-foreground text-lg">
              Master the concepts behind modern AI systems like ChatGPT
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                title: 'Softmax & Cross-Entropy',
                description: 'Understand probability distributions and multi-class loss functions',
                href: '/softmax',
                concepts: ['Softmax', 'Cross-Entropy', 'Gradients']
              },
              {
                title: 'Multi-Class Classification',
                description: 'Train neural networks to classify data into multiple categories',
                href: '/multiclass',
                concepts: ['Neural Networks', 'Training', 'Decision Boundaries']
              },
              {
                title: 'Attention Mechanism',
                description: 'Explore how models learn to focus on relevant tokens',
                href: '/attention',
                concepts: ['Self-Attention', 'Query-Key-Value', 'Attention Weights']
              },
              {
                title: 'Transformer Block',
                description: 'Build the fundamental building block of modern LLMs',
                href: '/transformer',
                concepts: ['Layer Norm', 'Feed-Forward', 'Residual Connections']
              },
              {
                title: 'Next-Token Prediction',
                description: 'See how language models generate text one token at a time',
                href: '/token-prediction',
                concepts: ['Autoregressive', 'Temperature', 'Sampling']
              },
              {
                title: 'GPT Decoder',
                description: 'Understand the complete architecture of GPT models',
                href: '/gpt-decoder',
                concepts: ['Stacked Layers', 'Causal Masking', 'Text Generation']
              }
            ].map((feature) => (
              <Card key={feature.href} className="border-2 hover:border-blue-500 transition-all group">
                <CardHeader>
                  <CardTitle className="group-hover:text-blue-600 transition-colors">
                    {feature.title}
                  </CardTitle>
                  <CardDescription>{feature.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-wrap gap-2">
                    {feature.concepts.map((concept) => (
                      <Badge key={concept} variant="secondary" className="text-xs">
                        {concept}
                      </Badge>
                    ))}
                  </div>
                  <Link href={feature.href}>
                    <Button className="w-full group-hover:bg-blue-600 transition-colors gap-2">
                      Explore
                      <ArrowRightIcon className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-gray-50 py-8">
        <div className="container mx-auto px-6 text-center text-sm text-muted-foreground">
          <p>Built with Next.js, React, and Shadcn UI</p>
          <p className="mt-2">Making ML internals intuitive, one visualization at a time.</p>
        </div>
      </footer>
    </div>
  );
}
