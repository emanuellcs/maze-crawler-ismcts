# Crawl: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Crawl competition on Kaggle.

## Game Overview

Crawl is a two-player real-time strategy game on a 20-wide maze that scrolls northward over time. Each player starts with a single **Factory** and must build robots to explore, collect energy, and outlast the opponent.

- **Factory** (indestructible) builds Scouts, Workers, and Miners; can JUMP every 20 turns
- **Scout** (cost 50) is fast with vision range 5 — your eyes
- **Worker** (cost 200) builds and removes walls (100 energy per action)
- **Miner** (cost 300) can `TRANSFORM` on a mining node into a mine that generates 50 energy/turn
- **Maze** has east/west symmetry with occasional doors; both players see only what their robots see (fog of war)
- **Combat**: when robots end the turn on the same cell, crush rules apply (Factory > Miner > Worker > Scout). Same-type collisions destroy all parties — friendly fire is real
- **Scrolling**: the southern boundary advances, destroying anything left behind. Speed ramps from 1/4 turns to 1/turn by step 400
- **Win condition**: last factory standing wins; if both survive to step 500, tiebreaker cascade is total energy → unit count → draw

See [README.md](README.md) for full rules and configuration defaults.

## Your Agent

Your agent is a function that receives an observation and configuration and returns a dict mapping robot UIDs to action strings.

**Observation fields:**

- `obs.player` — your player index (0 or 1)
- `obs.walls` — flat array of wall bitfields. Index = `(row - southBound) * width + col`. Value `-1` = undiscovered. Bits: `N=1, E=2, S=4, W=8`
- `obs.crystals` — `{"col,row": energy}`, only currently visible
- `obs.robots` — `{"uid": [type, col, row, energy, owner, move_cd, jump_cd, build_cd]}`. Types: `0=Factory, 1=Scout, 2=Worker, 3=Miner`
- `obs.mines` — `{"col,row": [energy, maxEnergy, owner]}`, remembered once seen
- `obs.miningNodes` — `{"col,row": 1}`, only currently visible
- `obs.southBound`, `obs.northBound` — current active row range

**Action format:**
Each value is an action string keyed by robot UID:

- Movement: `NORTH`, `SOUTH`, `EAST`, `WEST`, `IDLE`
- Factory: `BUILD_SCOUT`, `BUILD_WORKER`, `BUILD_MINER`, `JUMP_NORTH/SOUTH/EAST/WEST`
- Worker: `BUILD_NORTH/SOUTH/EAST/WEST`, `REMOVE_NORTH/SOUTH/EAST/WEST`
- Miner: `TRANSFORM` (must be on a mining node)
- Any robot: `TRANSFER_NORTH/SOUTH/EAST/WEST` to send all energy to an adjacent friendly

**Example — Build a Worker, March North:**

```python
from random import choice

def agent(obs, config):
    actions = {}
    width = config.width
    my_robots = {
        uid: data for uid, data in obs.robots.items()
        if data[4] == obs.player
    }

    for uid, data in my_robots.items():
        rtype, col, row, energy = data[0], data[1], data[2], data[3]
        build_cd = data[7] if len(data) > 7 else 0

        idx = (row - obs.southBound) * width + col
        w = obs.walls[idx] if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1 else 0

        if rtype == 0:  # Factory
            if w & 1:
                actions[uid] = "JUMP_NORTH"
            elif energy >= config.workerCost and build_cd == 0:
                actions[uid] = "BUILD_WORKER"
            else:
                actions[uid] = "NORTH"
        elif rtype == 2 and (w & 1) and energy >= config.wallRemoveCost:
            actions[uid] = "REMOVE_NORTH"
        else:
            passable = []
            if not (w & 1): passable.append("NORTH")
            if not (w & 2): passable.append("EAST")
            if not (w & 4): passable.append("SOUTH")
            if not (w & 8): passable.append("WEST")
            actions[uid] = "NORTH" if "NORTH" in passable else (choice(passable) if passable else "IDLE")

    return actions
```

## Test Locally

Install the environment from PyPI:

```bash
pip install kaggle-environments
```

Run a game from Python or a notebook:

```python
from kaggle_environments import make

env = make("crawl", configuration={"randomSeed": 42}, debug=True)
env.run(["main.py", "random"])

# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Render in a notebook
env.render(mode="ipython", width=800, height=800)
```

## Set Up the Kaggle CLI

Install the CLI:

```bash
pip install kaggle
```

You'll need a Kaggle account — sign up at https://www.kaggle.com if you don't have one. Then download your API credentials at https://www.kaggle.com/settings/api by clicking **"Generate New Token"** under the "API" section.

**Recommended: API token file.** Save the token string to `~/.kaggle/access_token`:

```bash
mkdir -p ~/.kaggle
# Paste the token from the Kaggle settings UI into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternative auth methods:

- **OAuth (browser flow):** `kaggle auth login`
- **Environment variable:** `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx`

Verify the CLI is wired up:

```bash
kaggle competitions list -s "crawl"
```

## Find the Competition

```bash
kaggle competitions list -s "crawl"
kaggle competitions pages crawl
kaggle competitions pages crawl --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/crawl` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download crawl -p crawl-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit crawl -f main.py -m "Worker rush v1"
```

**Multi-file agent** — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit crawl -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit crawl -k YOUR_USERNAME/crawl-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions crawl
```

Note the submission ID from the output — you'll need it for episodes.

## List Episodes

Once your submission has played some games:

```bash
kaggle competitions episodes <SUBMISSION_ID>
```

CSV output for scripting:

```bash
kaggle competitions episodes <SUBMISSION_ID> -v
```

## Download Replays and Logs

Download the replay JSON for an episode (for visualization or analysis):

```bash
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays
```

Download agent logs to debug your agent's behavior:

```bash
# Logs for the first agent (index 0)
kaggle competitions logs <EPISODE_ID> 0

# Logs for the second agent (index 1)
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

## Check the Leaderboard

```bash
kaggle competitions leaderboard crawl -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('crawl', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit crawl -f main.py -m "v1"

# Check status
kaggle competitions submissions crawl

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard crawl -s
```
