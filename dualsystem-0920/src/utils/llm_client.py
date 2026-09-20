"""Gemini LLM client for generating agent archetypes."""

import json
import os
import google.generativeai as genai
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from pathlib import Path
from config import CONFIG


_google_api_key = CONFIG.secrets.google_api_key
if not _google_api_key:
    raise ValueError(
        "GOOGLE_API_KEY is not set. Provide your Google (Gemini) API key via the "
        "GOOGLE_API_KEY environment variable before generating archetypes."
    )
genai.configure(api_key=_google_api_key, transport="rest")


MODEL_NAME = CONFIG.llm.gemini_model


class RawActivity(BaseModel):
    """An LLM-generated trip unit: one move followed by one stay."""
    time: str = Field(..., description="Departure Time: The time to START driving from the Origin (HH:MM).")

    origin_type: str = Field(..., description="Start Location Type: 'Home', 'Work', 'Relax', 'Other'")
    dest_type: str = Field(..., description="End Location Type: 'Home', 'Work', 'Relax', 'Other'")

    description: str = Field(..., description="Description of the trip purpose (e.g., 'Commute to work')")

    duration_min: int = Field(..., description="How long to stay at the DESTINATION after arrival (minutes)")


class AgentArchetype(BaseModel):
    """An agent archetype spanning three orthogonal experimental dimensions (2x2x2)."""
    role_type: Literal["COMMUTER", "GIG_WORKER"] = Field(..., description="Occupational role determining schedule constraint.")

    price_trait: Literal["SENSITIVE", "INSENSITIVE"] = Field(..., description="Trait: Sensitivity to charging price.")

    anxiety_trait: Literal["ANXIOUS", "CALM"] = Field(..., description="Trait: Anxiety towards low battery.")

    name_tag: str = Field(..., description="A short catchy name, e.g., 'The Panic Saver'")
    description: str = Field(..., description="A short persona summary combining these traits.")

    daily_routine: List[RawActivity] = Field(..., description="Daily schedule. MUST start with a dummy 'Wake Up' at Home.")


class ArchetypeList(BaseModel):
    archetypes: List[AgentArchetype]


class LLMFactory:
    def __init__(self):
        self.generation_config = {
            "response_mime_type": "application/json",
            "response_schema": ArchetypeList
        }
        self.model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=self.generation_config
        )

    def generate_archetypes(self) -> List[AgentArchetype]:
        """Generate exactly 8 archetypes for the 2x2x2 experimental design."""
        prompt = """
        You are a research scientist designing a Multi-Agent Simulation for Electric Vehicles.
        Your task is to generate exactly **8 distinct agent archetypes** corresponding to a 2x2x2 experimental design matrix.

        The 3 dimensions are:
        1. **Role**: COMMUTER (Fixed 9-5 schedule) vs GIG_WORKER (Flexible, high mileage).
        2. **Price Trait**: SENSITIVE (Frugal) vs INSENSITIVE (Wealthy/Urgent).
        3. **Anxiety Trait**: ANXIOUS (Risk-averse) vs CALM (Risk-taker).

        Please generate one archetype for EACH of the following 8 combinations:

        1. [COMMUTER, SENSITIVE, ANXIOUS]: The "Worried Saver". Fixed job, but constantly checks battery and looks for cheap slow chargers.
        2. [COMMUTER, SENSITIVE, CALM]: The "Strategic Optimizer". Fixed job, waits until the last minute to charge at the cheapest rate.
        3. [COMMUTER, INSENSITIVE, ANXIOUS]: The "Safety First". Fixed job, charges immediately upon arrival regardless of cost.
        4. [COMMUTER, INSENSITIVE, CALM]: The "Efficient Elite". Fixed job, only charges when necessary, prefers fast charging.
        5. [GIG_WORKER, SENSITIVE, ANXIOUS]: The "Grinding Panic". Drives all day, stressed about range, hunts for cheapest electrons.
        6. [GIG_WORKER, SENSITIVE, CALM]: The "Profit Maximizer". Drives until 5% battery, then finds the absolute cheapest spot.
        7. [GIG_WORKER, INSENSITIVE, ANXIOUS]: The "High-End Driver". High income, tops up constantly at Fast Chargers to avoid downtime.
        8. [GIG_WORKER, INSENSITIVE, CALM]: The "Volume Runner". Just drives. Charges at nearest fast charger only when empty.

        **CRITICAL Logic for 'daily_routine':**
        Each item in the routine is a **TRIP**.
        - `time`: When to START the car and leave `origin_type`.
        - `origin_type`: Where the car is currently parked.
        - `dest_type`: Where the car is going.
        - **Continuity Rule**: The `origin_type` of item N must match the `dest_type` of item N-1 (logically).
        - **Start/End**: The first trip must start at 'Home', and the last trip must end at 'Home'.

        **Schedule Requirements (Crucial):**
        - **Activity.time** represents **DEPARTURE TIME** (When to start the car).
        - **COMMUTER**:
            - **COMMUTER**: Must have 'Home' -> 'Work' (8h) -> 'Home'.
        - **GIG_WORKER**:
            - Needs at least 7-8 trips (Home -> Work -> Work ... -> Home).
            - Spaced out departure times.
        - Ensure routine starts and ends at **Home**.

        **CRITICAL Rules for 'daily_routine':**
        1. **Synchronize Major Activities**: Ensure 'COMMUTER' starts around 08:00-09:00 and ends around 17:00-18:00 for Commuters to create congestion.
        2. **Chronological Order**: Times MUST increase from morning to night (e.g., 07:00 -> 08:30 -> 12:00 -> 18:00).
        3. **Single Day**: All times must be between 00:00 and 23:59 of the SAME day. Do NOT cross midnight (e.g., no 01:00 after 23:00).
        4. **Start/End**: Must start at 'Home' in the morning and end at 'Home' in the evening.
        5. **Format**: 'time' is HH:MM (24h format)
        Output valid JSON matching the schema.
         **JSON Output Template (Strictly follow this structure):**
        {{
            "archetypes": [
                {{
                    "role_type": "COMMUTER",
                    "price_trait": "SENSITIVE",
                    "anxiety_trait": "ANXIOUS",
                    "name_tag": "The Worried Saver",
                    "description": "...",
                    "daily_routine": [
                        {{
                            {"time": "HH:MM (e.g., '07:30')",
                            "origin_type": "String (Must be one of: 'Home', 'Work', 'Relax', 'Other')",
                            "dest_type": "String (Must be one of: 'Home', 'Work', 'Relax', 'Other')",
                            "description": "String",
                            "duration_min": Integer
                        }},
                            .. (more activities to cover the day)
                    ]
                }},
                ... (7 more items)
            ]
        }}
        """

        print(f"🧠 [Gemini] Generating 8 experiment archetypes (2x2x2 matrix)...")

        try:
            response = self.model.generate_content(prompt)
            json_str = response.text

            data = json.loads(json_str)

            if isinstance(data, list):
                result_obj = ArchetypeList(archetypes=data)
            elif "archetypes" in data:
                result_obj = ArchetypeList(archetypes=data["archetypes"])
            else:
                result_obj = ArchetypeList(**data)

            if len(result_obj.archetypes) != 8:
                print(f"⚠️ Warning: LLM generated {len(result_obj.archetypes)} archetypes, expected 8.")

            return result_obj.archetypes

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return []


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    case_dir = project_root / "case"

    case_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_dir / "archetypes.json"

    print(f"🚀 Running LLM Client standalone...")
    print(f"📂 Target path: {output_path}")

    try:
        factory = LLMFactory()
        archetypes = factory.generate_archetypes()

        if archetypes:
            data = [a.model_dump(mode='json') for a in archetypes]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Success! File saved to: {output_path}")
            print(f"📊 Generated {len(archetypes)} archetypes.")
        else:
            print("❌ Generation result is empty.")

    except Exception as e:
        print(f"❌ Runtime error: {e}")
        print("💡 Hint: check whether GOOGLE_API_KEY is configured.")
