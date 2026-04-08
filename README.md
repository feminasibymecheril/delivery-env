# AI Logistics Optimization Environment

## Overview

This project simulates a real-world logistics optimization problem inspired by warehouse automation and last-mile delivery systems.

An AI agent must efficiently navigate an environment, pick up packages, and deliver them to target locations while minimizing cost and maximizing efficiency.

---

## Real-World Relevance

This environment represents challenges in:

* Warehouse robotics
* Delivery route optimization
* Autonomous logistics systems
* Resource-constrained planning

---

## Environment Details

* Grid size: 5x5
* Random positions every episode
* Includes obstacles, packages, and delivery goals

---

## Action Space

| Action | Description     |
| ------ | --------------- |
| 0      | Move Up         |
| 1      | Move Down       |
| 2      | Move Left       |
| 3      | Move Right      |
| 4      | Pick Package    |
| 5      | Deliver Package |

---

## Observation Space

```json
{
  "robot": [x, y],
  "packages": [[x, y]],
  "goals": [[x, y]],
  "has_package": true,
  "current_package_idx": 0,
  "obstacles": [[x, y]]
}
```

---

## Tasks

* Easy: Reach package
* Medium: Pick package
* Hard: Deliver package

---

## Reward Design

* +1 for moving closer
* +10 for picking package
* +20 for delivery
* -0.05 per step
* -2 for obstacle hit

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run Server

```bash
uvicorn server.app:app --reload
```

---

## Run Inference

```bash
python inference.py
```

---

## Team

AIGEN³
