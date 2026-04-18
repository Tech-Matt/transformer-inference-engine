# Transformer Inference Engine — Project Guide for Claude

## Project Goal
A pure C implementation of a transformer inference engine, with the long-term goal of porting it to embedded systems (resource-constrained hardware).

## User Background
- Graduate student in embedded systems engineering
- Solid embedded systems domain knowledge
- Basic-to-intermediate C experience (still learning)
- Still learning about AI/ML concepts
- Wants to learn deeply — explain concepts thoroughly, draw connections to embedded systems knowledge where helpful

## Collaboration Style
- Opposite of "vibecoding": deliberate, principled, educational
- Flag tradeoffs in architectural decisions (especially memory, performance, portability)
- **Keep PLAN.md up to date**: when the user completes a task or phase, update PLAN.md immediately — mark tasks with `[x]` and update phase status labels (e.g. NEXT → IN PROGRESS → DONE)

## Key Technical Constraints (for embedded target)
- Pure C only (no C++, no external ML frameworks)
- Heap allocation used for now (noted in README), but this will need revisiting for embedded
- Must eventually run on resource-constrained hardware — keep memory layout, alignment, and allocation patterns in mind from the start
