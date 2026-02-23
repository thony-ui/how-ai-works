# 🧠 AI From First Principles — Interactive ML Playground

An interactive, frontend-only web application that teaches how modern AI systems work — starting from gradient descent and backpropagation, all the way to convolutional neural networks.

This project turns core machine learning concepts into **step-by-step visual simulations**, allowing users to interact with weights, gradients, and filters in real time.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🎯 Project Motivation

Most ML education tools:

- Show static diagrams
- Skip intermediate computation steps
- Abstract away the math behind libraries

This project acts like a **debugger for neural networks**.

Users can:

- Step through forward passes
- Inspect gradients during backprop
- Visualize how convolution kernels transform images
- Compare manual math to PyTorch implementations

The goal is to build intuition from first principles — not just API familiarity.

## 🧩 Features (Phase 1 MVP)

### 1️⃣ Gradient Descent on a Line

**Concept:** Linear regression with Mean Squared Error.

**Interactive Features:**

- Drag data points on a 2D canvas
- Adjust learning rate slider
- Step through gradient descent updates
- Visualize:
  - Current prediction line
  - Loss curve over time
  - Parameter updates (w, b)

**What Users Learn:**

- How gradients are computed
- Why learning rate matters
- How loss decreases over iterations
- What convergence looks like

### 2️⃣ Backpropagation in a 2-Layer Neural Network

**Concept:** Fully connected network: Input → Hidden (ReLU) → Output → MSE Loss

**Interactive Features:**

- Step-by-step execution through:
  - Forward pass
  - Loss computation
  - Backward pass
  - Weight updates
- Visual network diagram with:
  - Activation values
  - Gradient flow visualization
  - Weight magnitudes

**What Users Learn:**

- Chain rule in action
- How gradients flow backward
- Why ReLU affects gradient propagation
- How weight updates reduce error

### 3️⃣ Convolution Playground

**Concept:** 2D convolution over an image grid.

**Interactive Features:**

- 5×5 image grid editor (draw your own patterns)
- 3×3 filter editor (customizable)
- Step-through convolution process showing:
  - Current sliding window
  - Dot product calculation
  - Feature map output
- Preset filters:
  - Edge detection
  - Blur
  - Sharpen
  - Emboss
  - Horizontal/Vertical edge detection

**What Users Learn:**

- What convolution really computes
- Why filters detect patterns
- How feature maps are generated
- Intuition behind CNN layers

## 🏗 Tech Stack

- **Frontend:** Next.js 16 + React 19
- **UI Components:** Shadcn UI (built on Radix UI)
- **Styling:** TailwindCSS v4
- **Rendering:** HTML5 Canvas for visualizations
- **State Management:** React hooks
- **Math Engine:** Custom lightweight matrix operations
- **TypeScript:** Full type safety throughout

**No backend required.** All simulations run client-side.

## 🧠 Design Philosophy

This project follows three principles:

1. **Transparency**  
   Every intermediate value is visible.

2. **Step-by-Step Execution**  
   Users can move forward one computation at a time.

3. **Dual View**
   - Interactive visual intuition
   - Exact PyTorch code beside it

This bridges theory and real-world ML implementation.

## 📂 Project Structure

```
how-ai-works/
├── app/
│   ├── page.tsx                    # Landing page
│   ├── layout.tsx                  # Root layout with navigation
│   ├── gradient-descent/page.tsx   # Gradient descent route
│   ├── backpropagation/page.tsx    # Backprop route
│   └── convolution/page.tsx        # Convolution route
├── components/
│   ├── Navigation.tsx              # Main navigation bar
│   ├── gradient-descent/           # Gradient descent components
│   ├── backprop/                   # Backpropagation components
│   ├── convolution/                # Convolution components
│   └── ui/                         # Shadcn UI components
├── lib/
│   ├── math/                       # Mathematical operations
│   │   ├── matrix.ts              # Matrix operations
│   │   ├── activations.ts         # Activation functions
│   │   ├── loss.ts                # Loss functions
│   │   ├── gradient.ts            # Gradient computations
│   │   ├── convolution.ts         # Convolution operations
│   │   └── network.ts             # Neural network class
│   └── utils.ts                   # Utility functions
└── public/                        # Static assets
```

## 🎨 Architecture & SOLID Principles

The codebase follows SOLID principles for maintainability:

- **Single Responsibility:** Each component/module has one clear purpose
- **Open/Closed:** Easy to extend with new playgrounds
- **Liskov Substitution:** Consistent interfaces across components
- **Interface Segregation:** Components receive only what they need
- **Dependency Inversion:** Depend on abstractions

## 📚 Educational Goals

By the end of the MVP, a user should:

- Understand gradient descent beyond formulas
- Be able to manually trace backpropagation
- Know what convolution computes at each step
- Connect mathematical intuition to PyTorch implementation

## 🔮 Roadmap (Future Phases)

- [ ] Softmax + Cross-Entropy visualization
- [ ] Multi-class classification demo
- [ ] Attention mechanism playground
- [ ] Mini Transformer block simulator
- [ ] Next-token prediction visualizer
- [ ] Interactive GPT decoder simulation

## 🏁 Why This Project Matters

This is not another "ML model trained on MNIST."

This is an educational tool designed to:

- Strengthen conceptual understanding
- Make ML internals intuitive
- Bridge theory and practice
- Help students think like ML engineers

---

**Built with ❤️ for ML education**

Making neural network internals intuitive, one visualization at a time.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
