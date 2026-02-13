"""
Prompt Templates Module

Structured prompt templates following methodology from research papers:
- Dörnbach et al.: 3-part prompts (Problem + Task + Format) for repurposing
- Sonntag et al.: 5-part prompts (Background + Problem + Context + Task + Format) for ML recommendation

This ensures consistent, reproducible prompt engineering across experiments.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


# ================= Safe Dict for Rendering =================

class SafeDict(dict):
    """
    Dictionary that returns the key itself if not found, preventing KeyError.
    Allows literal braces in templates.
    """
    def __missing__(self, key):
        return '{' + key + '}'


# ================= Prompt Template Base Class =================

@dataclass
class PromptTemplate:
    """
    Base class for structured prompts.

    Following research methodology, prompts have clear sections:
    - Problem Description: What challenge we're addressing
    - Task Description: What the LLM should do
    - Format Description: How to structure the output

    Optional sections:
    - Background Information: Domain context
    - Context: Retrieved knowledge (for RAG)

    Metadata:
    - paper_compliant: Whether this template follows published methodology exactly
    """

    problem_description: str
    task_description: str
    format_description: str
    background_information: Optional[str] = None
    context_template: Optional[str] = None  # For RAG prompts
    paper_compliant: bool = False  # Default to non-compliant for safety

    def render(self, **kwargs) -> str:
        """
        Render template with variables.

        Args:
            **kwargs: Variables to fill in the template

        Returns:
            Rendered prompt as string

        Example:
            template.render(fault_description="Error E18", appliance="Bosch")
        """
        sections = []

        # Extract context early to prevent it from being used in other sections
        context_value = kwargs.get("context", None)
        kwargs_without_context = {k: v for k, v in kwargs.items() if k != "context"}
        safe_dict = SafeDict(kwargs_without_context)

        # Background (if provided)
        if self.background_information:
            sections.append(self.background_information.format_map(safe_dict))

        # Problem
        sections.append(self.problem_description.format_map(safe_dict))

        # Context (if template has context_template)
        # IMPORTANT: Context must come BEFORE Task (Sonntag methodology)
        # For Sonntag 5-part structure, always render context if template exists
        if self.context_template:
            if context_value is not None:
                # Context template uses special handling to prevent injection
                context_section = self.context_template.replace("{context}", str(context_value))
                sections.append(context_section)
            else:
                # Render empty context section for 5-part structure compliance
                context_section = self.context_template.replace("{context}", "")
                sections.append(context_section)

        # Task
        sections.append(self.task_description.format_map(safe_dict))

        # Format
        sections.append(self.format_description.format_map(safe_dict))

        return "\n\n".join(sections)

    def __repr__(self) -> str:
        """String representation"""
        return f"PromptTemplate(sections={3 + bool(self.background_information) + bool(self.context_template)})"


# ================= Prompt Library =================

class PromptLibrary:
    """
    Pre-built prompt templates for different tasks.

    Templates are organized by task type and follow published methodologies.
    """

    # ==================== DIAGNOSIS PROMPTS ====================

    DIAGNOSIS_BASELINE = PromptTemplate(
        background_information="You are a helpful repair technician assistant.",
        problem_description=(
            "The user needs help diagnosing a fault with their household appliance."
        ),
        task_description=(
            "Diagnose the following fault: {fault_description}\n\n"
            "Appliance: {appliance}"
        ),
        format_description=(
            "Format your answer using the following structure:\n\n"
            "**Diagnosis:** [Identify the root cause - name the specific failed component or technical fault]\n"
            "**Recommended Action:** [Step-by-step fix]\n"
            "**Tools & Parts Required:** [List of tools and parts]\n"
            "**Safety Warnings:** [Important safety notes as blockquotes using >]"
        )
    )

    DIAGNOSIS_RAG = PromptTemplate(
        background_information=(
            "You are a repair assistant. You MUST answer STRICTLY using ONLY the provided manual. "
            "NEVER use general knowledge or information outside the manual."
        ),
        problem_description=(
            "The user needs help diagnosing a fault based on the repair manual provided below."
        ),
        context_template=(
            "=== REPAIR MANUAL (USE ONLY THIS INFORMATION) ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF MANUAL ==="
        ),
        task_description=(
            "Diagnose the following fault using ONLY the manual above:\n\n"
            "Fault: {fault_description}\n"
            "Appliance: {appliance}\n\n"
            "CRITICAL RULES:\n"
            "- Use ONLY information from the provided manual\n"
            "- Add [1], [2], [3] citation after EVERY factual statement\n"
            "- ONLY cite chunk numbers that appear in the manual context above\n"
            "- NEVER guess or invent citation numbers\n"
            "- If ANY information is not in the manual, respond EXACTLY: 'Not found in the manual.'"
        ),
        format_description=(
            "Format:\n"
            "**Diagnosis:** [Root cause - identify the specific failed component or technical fault] [1]\n"
            "**Recommended Action:** [Step-by-step fix with citations] [2]\n"
            "**Tools & Parts Required:** [List] [3]\n"
            "**Safety Warnings:** [Important notes] [4]\n\n"
            "REMEMBER: Every factual statement MUST have [N] citation. "
            "ONLY cite chunks from the manual. "
            "If not found, write: 'Not found in the manual.'"
        )
    )

    # ==================== DIAGNOSIS PROMPT VARIATIONS ====================

    DIAGNOSIS_BASELINE_V1 = PromptTemplate(
        background_information="You are an expert appliance repair technician.",
        problem_description=(
            "A customer has reported a fault with their household appliance. "
            "Your expertise is needed to provide accurate diagnosis and repair guidance."
        ),
        task_description=(
            "Fault reported: {fault_description}\n\n"
            "Appliance type: {appliance}\n\n"
            "Provide a comprehensive diagnosis with detailed repair instructions."
        ),
        format_description=(
            "Structure your response as follows:\n\n"
            "**1. Diagnosis:**\n"
            "[Detailed explanation of what is wrong and why]\n\n"
            "**2. Root Cause:**\n"
            "[Underlying cause of the fault]\n\n"
            "**3. Repair Procedure:**\n"
            "[Detailed step-by-step instructions]\n\n"
            "**4. Required Materials:**\n"
            "- Tools: [List specific tools]\n"
            "- Parts: [List replacement parts with specifications]\n\n"
            "**5. Safety Precautions:**\n"
            "> [Critical safety warnings as blockquotes]\n\n"
            "**6. Estimated Difficulty:**\n"
            "[Beginner/Intermediate/Advanced] - [Time estimate]"
        )
    )

    DIAGNOSIS_BASELINE_CONCISE = PromptTemplate(
        background_information="You are a repair technician assistant.",
        problem_description="The user needs quick help diagnosing an appliance fault.",
        task_description=(
            "Fault: {fault_description}\n"
            "Appliance: {appliance}"
        ),
        format_description=(
            "Provide a brief, focused response:\n\n"
            "**Diagnosis:** [Root cause - specific failed component]\n"
            "**Fix:** [Concise steps]\n"
            "**Need:** [Tools & parts]\n"
            "**Warning:** [Key safety note]"
        )
    )

    DIAGNOSIS_BASELINE_SAFETY_FOCUSED = PromptTemplate(
        background_information=(
            "You are a safety-conscious repair technician. "
            "User safety is your top priority in all repair guidance."
        ),
        problem_description=(
            "The user needs help with an appliance fault. "
            "Many appliances involve electrical hazards, so safety guidance is critical."
        ),
        task_description=(
            "Diagnose and provide repair guidance for:\n\n"
            "Fault: {fault_description}\n"
            "Appliance: {appliance}\n\n"
            "EMPHASIZE SAFETY at every step."
        ),
        format_description=(
            "**SAFETY FIRST:**\n"
            "> [Critical warnings - electrical, mechanical, chemical hazards]\n"
            "> [Required safety equipment]\n\n"
            "**Diagnosis:**\n"
            "[Root cause - identify the specific failed component or technical fault]\n\n"
            "**Safe Repair Procedure:**\n"
            "1. [Step with safety notes]\n"
            "2. [Step with safety notes]\n"
            "3. [Step with safety notes]\n\n"
            "**Tools & Parts:**\n"
            "[List with safety specifications]\n\n"
            "**When to Call a Professional:**\n"
            "[Situations that require expert help]"
        )
    )

    DIAGNOSIS_RAG_V1 = PromptTemplate(
        background_information=(
            "You are a repair assistant with access to the official service manual. "
            "Your job is to extract relevant information from the manual and cite your sources accurately. "
            "NEVER use information outside the provided manual."
        ),
        problem_description=(
            "The user has a fault and you have access to the service manual below. "
            "Provide accurate diagnosis based solely on the manual content."
        ),
        context_template=(
            "=== SERVICE MANUAL (USE ONLY THIS INFORMATION) ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF MANUAL ==="
        ),
        task_description=(
            "Using ONLY the provided service manual, diagnose:\n\n"
            "Fault: {fault_description}\n"
            "Appliance: {appliance}\n\n"
            "Search the manual thoroughly for relevant information.\n\n"
            "CRITICAL RULES:\n"
            "- Use ONLY information from the manual above\n"
            "- Add [1], [2], [3] citation after EVERY factual statement\n"
            "- ONLY cite chunk numbers that appear in the manual context above\n"
            "- NEVER guess or invent citation numbers\n"
            "- If information spans multiple chunks, cite all: [1], [2], [3]\n"
            "- If not in manual, state EXACTLY: 'This information is not available in the provided manual.'"
        ),
        format_description=(
            "RESPONSE FORMAT:\n\n"
            "**Diagnosis:** [Root cause - identify the specific failed component or technical fault] [1]\n\n"
            "**Recommended Repair Steps:**\n"
            "1. [Step one] [2]\n"
            "2. [Step two] [2]\n"
            "3. [Step three] [3]\n\n"
            "**Required Tools & Parts:** [List] [4]\n\n"
            "**Safety Warnings:** [Warnings from manual] [5]\n\n"
            "**Additional Notes:** [Any relevant context] [6]\n\n"
            "REMEMBER: Every factual claim MUST include [N] citation. "
            "ONLY cite chunks from the manual. "
            "If not found, write: 'This information is not available in the provided manual.'"
        )
    )

    DIAGNOSIS_RAG_CONCISE = PromptTemplate(
        background_information=(
            "You are a repair assistant. Use ONLY the provided manual. Cite page numbers. "
            "NEVER use external knowledge."
        ),
        problem_description="Extract concise diagnosis from the service manual below.",
        context_template=(
            "=== SERVICE MANUAL ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF MANUAL ==="
        ),
        task_description=(
            "Using ONLY the manual above:\n\n"
            "Fault: {fault_description}\n"
            "Appliance: {appliance}\n\n"
            "RULES: Use only manual content. Add [1], [2], [3] after every claim. "
            "ONLY cite chunk numbers from the manual. NEVER invent citation numbers. "
            "If not found, write: 'Not found in manual.'"
        ),
        format_description=(
            "Brief response with citations:\n\n"
            "**Diagnosis:** [Root cause - specific failed component] [1]\n"
            "**Solution:** [Quick fix] [2]\n"
            "**Parts:** [List] [3]\n\n"
            "ONLY cite chunks from manual. If not in manual: 'Not found in manual.'"
        )
    )

    DIAGNOSIS_RAG_SAFETY_FOCUSED = PromptTemplate(
        background_information=(
            "You are a safety-conscious repair assistant with access to the official service manual. "
            "User safety is your top priority. Every recommendation must be cited from the manual. "
            "NEVER use information outside the manual."
        ),
        problem_description=(
            "The user needs help with an appliance fault. Many appliances involve electrical hazards. "
            "Provide diagnosis based strictly on the manual below, emphasizing safety at every step."
        ),
        context_template=(
            "=== SERVICE MANUAL (SAFETY INFORMATION IS CRITICAL) ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF MANUAL ==="
        ),
        task_description=(
            "Using ONLY the provided service manual, diagnose:\n\n"
            "Fault: {fault_description}\n"
            "Appliance: {appliance}\n\n"
            "EMPHASIZE SAFETY. Cite all safety warnings from the manual.\n\n"
            "CRITICAL RULES:\n"
            "- Every safety warning MUST include [N] citation\n"
            "- ONLY cite chunk numbers that appear in the manual context above\n"
            "- NEVER guess or invent citation numbers\n"
            "- Use ONLY information from the manual above\n"
            "- If safety info not in manual, state: 'Safety information not available in manual.'"
        ),
        format_description=(
            "FORMAT:\n\n"
            "**SAFETY FIRST:**\n"
            "> [Critical warnings from manual] [1]\n"
            "> [Required safety equipment] [2]\n"
            "> [Electrical hazards] [3]\n\n"
            "**Diagnosis:** [Root cause - identify the specific failed component] [4]\n\n"
            "**Safe Repair Steps:**\n"
            "1. [Step with safety notes] [5]\n"
            "2. [Step with safety notes] [6]\n\n"
            "**Tools & Parts:** [List] [7]\n\n"
            "**When to Call a Professional:**\n"
            "[Situations requiring expert help] [8]\n\n"
            "REMEMBER: ONLY cite chunks from the manual. Cite every safety claim with [N]."
        )
    )

    # ==================== REPURPOSING PROMPTS (Dörnbach) ====================
    # METHODOLOGY: Dörnbach paper requires exactly 3 parts: Problem, Task, Format
    # RQ1 (scenarios): Output exactly 10 lines, format "Component | Target System"
    # RQ2 (properties): Output all properties, one per line, no limit
    #
    # PAPER-COMPLIANT TEMPLATES (3 parts only, NO background):
    # - REPURPOSING (baseline) ✅
    # - REPURPOSING_PROPERTIES ✅
    #
    # NON-PAPER-COMPLIANT (custom experimental variants with background):
    # - REPURPOSING_DETAILED, REPURPOSING_CREATIVE, REPURPOSING_TECHNICAL, REPURPOSING_CONCISE
    #
    # CUSTOM RAG EXTENSIONS (NOT Dörnbach methodology):
    # - All REPURPOSING_RAG* variants use 5-part structure (B+P+C+T+F)

    REPURPOSING = PromptTemplate(
        background_information=None,  # Dörnbach: NO background in default template
        problem_description=(
            "Many products are discarded while still functional. "
            "Identifying technically feasible repurposing scenarios can extend product lifecycles "
            "and reduce waste."
        ),
        task_description=(
            "Identify exactly 10 technically feasible repurposing scenarios for the following component:\n\n"
            "{component}\n\n"
            "Consider technical specifications, required modifications, and practical applications.\n\n"
            "IMPORTANT: Generate exactly 10 scenarios."
        ),
        format_description=(
            "Output EXACTLY 10 lines in the following format (no numbering, no explanations, no extra text):\n\n"
            "Component | Target System\n\n"
            "Example:\n"
            "Laptop screen | Digital picture frame\n"
            "Car battery | Home backup power system\n\n"
            "Output exactly 10 lines in Component | Target System format."
        ),
        paper_compliant=True  # Dörnbach 3-part methodology (Problem + Task + Format)
    )

    # ==================== REPURPOSING PROMPT VARIATIONS ====================
    # WARNING: These variants include background_information, making them NON-paper-compliant
    # They are custom experimental variants for comparing prompt designs
    # ALL variants must produce exactly 10 scenarios in Component | Target System format

    REPURPOSING_DETAILED = PromptTemplate(
        background_information=(
            "You are an expert in sustainable product design and circular economy. "
            "Your role is to identify repurposing opportunities that extend product lifecycles."
        ),
        problem_description=(
            "Many functional components are discarded unnecessarily. "
            "By identifying creative repurposing opportunities, we can reduce waste and create value."
        ),
        task_description=(
            "Identify exactly 10 repurposing scenarios for:\n\n"
            "Component: {component}\n\n"
            "For each scenario, consider:\n"
            "1. Technical compatibility and specifications\n"
            "2. Required modifications (prefer minimal changes)\n"
            "3. Practical application and market demand\n"
            "4. Economic viability\n"
            "5. Environmental impact\n\n"
            "IMPORTANT: Generate exactly 10 scenarios, each starting with Component | Target System format."
        ),
        format_description=(
            "Format each scenario with detailed justification:\n\n"
            "**Scenario [N]: Component | Target System**\n"
            "- **Application:** [How it would be used]\n"
            "- **Feasibility:** [Technical compatibility]\n"
            "- **Modifications:** [What changes are needed]\n"
            "- **Value:** [Benefits and applications]\n\n"
            "Provide exactly 10 detailed scenarios."
        )
    )

    REPURPOSING_CREATIVE = PromptTemplate(
        background_information=(
            "You are an innovative designer specializing in creative reuse and upcycling. "
            "Think outside the box while maintaining technical feasibility."
        ),
        problem_description=(
            "The most valuable repurposing ideas often come from creative combinations "
            "and applications that aren't immediately obvious."
        ),
        task_description=(
            "Brainstorm exactly 10 creative repurposing scenarios for:\n\n"
            "{component}\n\n"
            "Think creatively:\n"
            "- Unconventional applications\n"
            "- Cross-domain repurposing\n"
            "- Artistic or educational uses\n"
            "- Smart home and IoT integration\n"
            "- Community and social projects\n\n"
            "IMPORTANT: Generate exactly 10 scenarios in Component | Target System format."
        ),
        format_description=(
            "Output exactly 10 lines in format:\n"
            "Component | Target System | Creative Angle\n\n"
            "Example:\n"
            "Washing machine drum | Fire pit | Artistic outdoor feature\n"
            "Old laptop webcam | Wildlife camera | Conservation project\n\n"
            "Provide exactly 10 scenarios."
        )
    )

    REPURPOSING_TECHNICAL = PromptTemplate(
        background_information=(
            "You are a technical engineer specializing in component reuse and system integration."
        ),
        problem_description=(
            "Identifying technically sound repurposing scenarios requires detailed specifications "
            "and compatibility analysis."
        ),
        task_description=(
            "Analyze the component and identify exactly 10 repurposing scenarios:\n\n"
            "{component}\n\n"
            "Focus on technical aspects:\n"
            "- Electrical/mechanical specifications\n"
            "- Interface compatibility (connectors, protocols)\n"
            "- Power requirements and efficiency\n"
            "- Physical dimensions and mounting\n"
            "- Operating conditions and limitations\n\n"
            "IMPORTANT: Generate exactly 10 scenarios, each starting with Component | Target System format."
        ),
        format_description=(
            "Format with technical specifications:\n\n"
            "Component | Target System\n"
            "Technical Requirements: [Specs needed]\n"
            "Compatibility: [Interfaces/protocols]\n"
            "Modifications: [Technical changes]\n\n"
            "Provide exactly 10 technically-detailed scenarios."
        )
    )

    REPURPOSING_CONCISE = PromptTemplate(
        background_information="You are a repurposing consultant providing quick scenario assessments.",
        problem_description="Identify practical repurposing opportunities efficiently.",
        task_description=(
            "List exactly 10 repurposing scenarios for: {component}\n\n"
            "Focus on practical, feasible options.\n\n"
            "IMPORTANT: Generate exactly 10 scenarios."
        ),
        format_description=(
            "Output EXACTLY 10 lines in format (no numbering, no extra text):\n"
            "Component | Target System\n\n"
            "Example:\n"
            "Laptop screen | Digital signage\n"
            "Car battery | Solar storage\n\n"
            "Provide exactly 10 lines in Component | Target System format."
        )
    )

    # ==================== REPURPOSING RAG VARIANTS ====================
    # CUSTOM RAG EXTENSION (NOT Dörnbach methodology)
    # These use 5-part structure (Background + Problem + Context + Task + Format)
    # Inspired by GraphRAG approach, not part of Dörnbach paper
    # ALL RAG variants must produce exactly 10 scenarios with page citations
    #
    # CRITICAL: Page citations require your RAG chunking pipeline to inject page markers
    # into the retrieved context, e.g., "[p.12] content [p.13] more content"
    # Without page markers in {context}, the model CANNOT cite accurately

    REPURPOSING_RAG = PromptTemplate(
        background_information=(
            "You are a repurposing expert with access to technical documentation. "
            "Use ONLY information from the provided documents. Cite page numbers."
        ),
        problem_description=(
            "Identify repurposing scenarios based strictly on the technical specifications "
            "and capabilities documented in the provided materials below."
        ),
        context_template=(
            "=== TECHNICAL DOCUMENTATION ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF DOCUMENTATION ==="
        ),
        task_description=(
            "Based ONLY on the provided technical documentation, identify exactly 10 repurposing scenarios for:\n\n"
            "{component}\n\n"
            "Use specifications from the documents to ensure technical feasibility.\n\n"
            "IMPORTANT: Generate exactly 10 scenarios.\n"
            "RULE: ONLY cite page numbers that actually appear in the documentation. NEVER invent page numbers."
        ),
        format_description=(
            "Output EXACTLY 10 lines in format (no numbering, no explanations):\n"
            "Component | Target System (p. X)\n\n"
            "Example:\n"
            "Laptop screen | Digital signage (p. 12)\n"
            "Car battery | Solar storage (p. 8)\n\n"
            "Provide exactly 10 scenarios. ONLY cite pages from the documentation."
        )
    )

    REPURPOSING_RAG_DETAILED = PromptTemplate(
        background_information=(
            "You are an expert in sustainable design with access to technical documentation. "
            "Analyze specifications thoroughly and cite all sources."
        ),
        problem_description=(
            "Identify comprehensive repurposing scenarios using detailed technical specifications "
            "from the provided documentation below."
        ),
        context_template=(
            "=== TECHNICAL DOCUMENTATION ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF DOCUMENTATION ==="
        ),
        task_description=(
            "Using ONLY the provided technical documentation, identify exactly 10 repurposing scenarios for:\n\n"
            "Component: {component}\n\n"
            "For each scenario, reference:\n"
            "1. Technical specifications from the docs\n"
            "2. Compatibility analysis\n"
            "3. Required modifications\n"
            "4. Feasibility assessment\n\n"
            "IMPORTANT: Generate exactly 10 scenarios, each starting with Component | Target System format.\n"
            "RULE: ONLY cite page numbers from the documentation. NEVER invent page numbers."
        ),
        format_description=(
            "Format each scenario with detailed justification:\n\n"
            "**Scenario [N]: Component | Target System**\n"
            "- **Specifications:** [From docs] (p. X)\n"
            "- **Application:** [Usage based on specs] (p. Y)\n"
            "- **Compatibility:** [Technical match] (p. Z)\n"
            "- **Modifications:** [Changes needed]\n"
            "- **Feasibility:** [Assessment based on docs]\n\n"
            "Provide exactly 10 detailed scenarios. ONLY cite pages from the documentation."
        )
    )

    REPURPOSING_RAG_CREATIVE = PromptTemplate(
        background_information=(
            "You are an innovative designer with access to technical specifications. "
            "Think creatively while ensuring technical validity from the documentation."
        ),
        problem_description=(
            "Find innovative repurposing opportunities grounded in documented specifications below."
        ),
        context_template=(
            "=== TECHNICAL SPECIFICATIONS ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF SPECIFICATIONS ==="
        ),
        task_description=(
            "Using the provided technical documentation, brainstorm exactly 10 creative repurposing scenarios for:\n\n"
            "{component}\n\n"
            "Balance creativity with technical feasibility based on documented specs.\n\n"
            "IMPORTANT: Generate exactly 10 scenarios in Component | Target System format.\n"
            "RULE: ONLY cite page numbers from the documentation. NEVER invent page numbers."
        ),
        format_description=(
            "Output exactly 10 lines in format:\n"
            "Component | Target System | Creative Angle (p. X)\n\n"
            "Example:\n"
            "Washing drum | Firepit | Heat specs allow artistic use (p. 15)\n\n"
            "Provide exactly 10 scenarios. ONLY cite pages from the documentation."
        )
    )

    REPURPOSING_RAG_TECHNICAL = PromptTemplate(
        background_information=(
            "You are a technical engineer with access to detailed component specifications."
        ),
        problem_description=(
            "Identify technically rigorous repurposing scenarios based on documented specifications below."
        ),
        context_template=(
            "=== TECHNICAL SPECIFICATIONS ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF SPECIFICATIONS ==="
        ),
        task_description=(
            "Using the technical documentation, analyze and identify exactly 10 repurposing scenarios for:\n\n"
            "{component}\n\n"
            "Focus on:\n"
            "- Exact specifications from docs\n"
            "- Interface compatibility\n"
            "- Operating parameters\n"
            "- Physical constraints\n\n"
            "IMPORTANT: Generate exactly 10 scenarios, each starting with Component | Target System format.\n"
            "RULE: ONLY cite page numbers from the documentation. NEVER invent page numbers."
        ),
        format_description=(
            "Format with technical specifications:\n\n"
            "Component | Target System\n"
            "Specs: [From documentation] (p. X)\n"
            "Voltage/Power: [Values] (p. Y)\n"
            "Interfaces: [Types] (p. Z)\n"
            "Compatibility: [Analysis]\n"
            "Modifications: [Technical details]\n\n"
            "Provide exactly 10 technically detailed scenarios. ONLY cite pages from the documentation."
        )
    )

    REPURPOSING_RAG_CONCISE = PromptTemplate(
        background_information=(
            "You are a repurposing consultant with access to technical specifications."
        ),
        problem_description="Quick repurposing scenarios validated by documentation below.",
        context_template=(
            "=== TECHNICAL DOCUMENTATION ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF DOCUMENTATION ==="
        ),
        task_description=(
            "From the provided documentation, list exactly 10 repurposing scenarios for:\n\n"
            "{component}\n\n"
            "IMPORTANT: Generate exactly 10 scenarios.\n"
            "RULE: ONLY cite page numbers from the documentation. NEVER invent page numbers."
        ),
        format_description=(
            "Output EXACTLY 10 lines in format (no numbering):\n"
            "Component | Target System (p. X)\n\n"
            "Provide exactly 10 scenarios. ONLY cite pages from the documentation."
        )
    )

    # RQ2: Property identification (Dörnbach methodology)
    # METHODOLOGY: Ask for "all relevant properties", no limit, one per line

    REPURPOSING_PROPERTIES = PromptTemplate(
        background_information=None,  # Dörnbach: NO background in default template
        problem_description=(
            "For successful repurposing, we need to identify all relevant technical properties "
            "of components that enable their reuse in different applications."
        ),
        task_description=(
            "List all relevant technical properties of:\n\n"
            "{component}\n\n"
            "Include:\n"
            "- Technical specifications (voltage, capacity, dimensions, materials, etc.)\n"
            "- Interfaces and connectors\n"
            "- Physical characteristics\n"
            "- Functional capabilities\n"
            "- Operating parameters\n"
            "- Any other properties relevant for repurposing"
        ),
        format_description=(
            "Output one property per line, using natural language descriptions.\n\n"
            "Example format:\n"
            "Voltage: 400V DC\n"
            "Capacity: 60kWh\n"
            "Interface: CAN bus communication\n"
            "Cooling: Liquid cooling system required\n"
            "Dimensions: 120cm x 80cm x 30cm\n"
            "Weight: 450kg\n\n"
            "List ALL relevant properties (no limit)."
        ),
        paper_compliant=True  # Dörnbach 3-part methodology (Problem + Task + Format) for RQ2
    )
    # ==================== ML RECOMMENDATION PROMPT VARIATIONS ===================
    # These prompts follow Sonntag et al. (2025) methodology exactly
    # Using 'problem_formulation' field for dataset compatibility

    ML_RECOMMENDATION_SONNTAG_BASELINE = PromptTemplate(
        background_information=(
            "You are an expert in both product development and machine learning. "
            "Your task is to recommend suitable ML algorithms for product development problems."
        ),
        problem_description=(
            "A product development team needs guidance on which machine learning approach "
            "would be most appropriate for their specific problem."
        ),
        task_description=(
            "Analyze the following product development problem:\n\n"
            "Problem: {problem_formulation}\n\n"
            "{background_information}\n\n"
            "Identify:\n"
            "1. The underlying ML problem type(s) (Classification, Regression, Clustering, Association Rule)\n"
            "2. Specific ML algorithm(s) that would be suitable for this problem"
        ),
        format_description=(
            "Format your response as:\n\n"
            "**ML Problem Type:** [Identify the problem type(s)]\n\n"
            "**Recommended Algorithm(s):** [List specific algorithms with brief justification]\n\n"
            "**Rationale:** [Explain why these algorithms are suitable]"
        ),
        paper_compliant=True
    )

    ML_RECOMMENDATION_SONNTAG_RAG = PromptTemplate(
        background_information=(
            "You are an expert in both product development and machine learning with access "
            "to a knowledge base of proven ML applications in product development. "
            "Use ONLY the provided knowledge to make your recommendations."
        ),
        problem_description=(
            "A product development team needs guidance on which machine learning approach "
            "would be most appropriate based on similar problems in the knowledge base."
        ),
        context_template=(
            "=== KNOWLEDGE BASE: ML APPLICATION EXAMPLES ===\n"
            "```\n"
            "{context}\n"
            "```\n"
            "=== END OF KNOWLEDGE BASE ==="
        ),
        task_description=(
            "Using ONLY the knowledge base above, analyze:\n\n"
            "Problem: {problem_formulation}\n\n"
            "{background_information}\n\n"
            "Find similar ML applications in the knowledge base and identify:\n"
            "1. The underlying ML problem type(s) based on similar cases [1], [2], etc.\n"
            "2. Specific ML algorithm(s) proven to work for similar problems [3], [4], etc.\n\n"
            "CRITICAL RULES:\n"
            "- Use ONLY information from the knowledge base\n"
            "- Add [1], [2], [3] citation after EVERY factual statement\n"
            "- ONLY cite chunk numbers that appear in the knowledge base above\n"
            "- NEVER guess or invent citation numbers\n"
            "- If not found in knowledge base, state: 'No similar case found in knowledge base.'"
        ),
        format_description=(
            "Format:\n"
            "**ML Problem Type:** [Type with citation] [1]\n\n"
            "**Recommended Algorithm(s):** [Algorithms with citations] [2], [3]\n\n"
            "**Rationale:** [Why suitable, based on knowledge base] [4]\n\n"
            "REMEMBER: Every claim must cite the knowledge base using [N] format. "
            "ONLY cite chunks from the knowledge base. "
            "If not found, write: 'No similar case found in knowledge base.'"
        ),
        paper_compliant=True
    )

    @staticmethod
    def get_ml_recommendation_template(use_rag: bool = False, variant: Optional[str] = None) -> PromptTemplate:
        """Get ML algorithm recommendation prompt template (Sonntag methodology)."""
        if variant and variant.lower() == "sonntag":
            # Sonntag-specific templates using problem_formulation
            if use_rag:
                return PromptLibrary.ML_RECOMMENDATION_SONNTAG_RAG
            else:
                return PromptLibrary.ML_RECOMMENDATION_SONNTAG_BASELINE
        else:
            # Default templates using problem_description
            if use_rag:
                return PromptLibrary.ML_RECOMMENDATION_RAG
            else:
                return PromptLibrary.ML_RECOMMENDATION

    # ==================== HELPER METHODS ====================

    @staticmethod
    def get_template(task_type: str, use_rag: bool = False, variant: Optional[str] = None) -> PromptTemplate:
        """
        Get appropriate template for a task type.

        Args:
            task_type: Type of task ("diagnosis", "repurposing", "repurposing_properties", "ml_recommendation")
            use_rag: Whether to use RAG version (if available)
            variant: Optional variant name (e.g., "v1", "concise", "safety_focused")

        Returns:
            PromptTemplate instance

        Raises:
            ValueError: If task_type or variant is unknown
        """
        if task_type == "diagnosis":
            # Handle diagnosis variants
            if variant:
                variant_lower = variant.lower()

                # RAG variants
                if use_rag:
                    if variant_lower == "v1":
                        return PromptLibrary.DIAGNOSIS_RAG_V1
                    elif variant_lower == "concise":
                        return PromptLibrary.DIAGNOSIS_RAG_CONCISE
                    elif variant_lower == "safety_focused" or variant_lower == "safety":
                        return PromptLibrary.DIAGNOSIS_RAG_SAFETY_FOCUSED
                    else:
                        raise ValueError(
                            f"Unknown diagnosis RAG variant: {variant}. "
                            f"Available: v1, concise, safety_focused"
                        )
                # Baseline variants
                else:
                    if variant_lower == "v1":
                        return PromptLibrary.DIAGNOSIS_BASELINE_V1
                    elif variant_lower == "concise":
                        return PromptLibrary.DIAGNOSIS_BASELINE_CONCISE
                    elif variant_lower == "safety_focused" or variant_lower == "safety":
                        return PromptLibrary.DIAGNOSIS_BASELINE_SAFETY_FOCUSED
                    else:
                        raise ValueError(
                            f"Unknown diagnosis baseline variant: {variant}. "
                            f"Available: v1, concise, safety_focused"
                        )
            # Default diagnosis prompts (no variant specified)
            return PromptLibrary.DIAGNOSIS_RAG if use_rag else PromptLibrary.DIAGNOSIS_BASELINE

        elif task_type == "repurposing":
            # Handle repurposing variants
            if variant:
                variant_lower = variant.lower()

                # RAG variants
                if use_rag:
                    if variant_lower == "detailed":
                        return PromptLibrary.REPURPOSING_RAG_DETAILED
                    elif variant_lower == "creative":
                        return PromptLibrary.REPURPOSING_RAG_CREATIVE
                    elif variant_lower == "technical":
                        return PromptLibrary.REPURPOSING_RAG_TECHNICAL
                    elif variant_lower == "concise":
                        return PromptLibrary.REPURPOSING_RAG_CONCISE
                    else:
                        raise ValueError(
                            f"Unknown repurposing RAG variant: {variant}. "
                            f"Available: detailed, creative, technical, concise"
                        )
                # Baseline variants
                else:
                    if variant_lower == "detailed":
                        return PromptLibrary.REPURPOSING_DETAILED
                    elif variant_lower == "creative":
                        return PromptLibrary.REPURPOSING_CREATIVE
                    elif variant_lower == "technical":
                        return PromptLibrary.REPURPOSING_TECHNICAL
                    elif variant_lower == "concise":
                        return PromptLibrary.REPURPOSING_CONCISE
                    else:
                        raise ValueError(
                            f"Unknown repurposing baseline variant: {variant}. "
                            f"Available: detailed, creative, technical, concise"
                        )
            # Default repurposing prompts (no variant specified)
            return PromptLibrary.REPURPOSING_RAG if use_rag else PromptLibrary.REPURPOSING

        elif task_type == "repurposing_properties":
            return PromptLibrary.REPURPOSING_PROPERTIES

        elif task_type == "ml_recommendation":
            # Handle ML recommendation variants
            if variant:
                variant_lower = variant.lower()

                # Sonntag validation variant (uses problem_formulation field)
                if variant_lower == "sonntag":
                    return PromptLibrary.get_ml_recommendation_template(use_rag, variant="sonntag")

                # RAG variants
                if use_rag:
                    if variant_lower == "detailed":
                        return PromptLibrary.ML_RECOMMENDATION_RAG_DETAILED
                    elif variant_lower == "beginner_friendly" or variant_lower == "beginner":
                        return PromptLibrary.ML_RECOMMENDATION_RAG_BEGINNER_FRIENDLY
                    elif variant_lower == "technical":
                        return PromptLibrary.ML_RECOMMENDATION_RAG_TECHNICAL
                    elif variant_lower == "concise":
                        return PromptLibrary.ML_RECOMMENDATION_RAG_CONCISE
                    else:
                        raise ValueError(
                            f"Unknown ml_recommendation RAG variant: {variant}. "
                            f"Available: detailed, beginner_friendly, technical, concise, sonntag"
                        )
                # Baseline variants
                else:
                    if variant_lower == "detailed":
                        return PromptLibrary.ML_RECOMMENDATION_DETAILED
                    elif variant_lower == "beginner_friendly" or variant_lower == "beginner":
                        return PromptLibrary.ML_RECOMMENDATION_BEGINNER_FRIENDLY
                    elif variant_lower == "technical":
                        return PromptLibrary.ML_RECOMMENDATION_TECHNICAL
                    elif variant_lower == "concise":
                        return PromptLibrary.ML_RECOMMENDATION_CONCISE
                    else:
                        raise ValueError(
                            f"Unknown ml_recommendation baseline variant: {variant}. "
                            f"Available: detailed, beginner_friendly, technical, concise, sonntag"
                        )
            # Default ML recommendation prompts (no variant specified)
            return PromptLibrary.ML_RECOMMENDATION_RAG if use_rag else PromptLibrary.ML_RECOMMENDATION

        else:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Supported: diagnosis, repurposing, repurposing_properties, ml_recommendation"
            )

    @staticmethod
    def list_available_templates() -> Dict[str, str]:
        """
        List all available templates with descriptions.

        Returns:
            Dictionary mapping template names to descriptions
        """
        return {
            "diagnosis (baseline)": "Fault diagnosis without manual (baseline)",
            "diagnosis (baseline, v1)": "Detailed diagnosis with comprehensive structure",
            "diagnosis (baseline, concise)": "Brief, focused diagnosis format",
            "diagnosis (baseline, safety_focused)": "Safety-first diagnosis approach",
            "diagnosis (RAG)": "Fault diagnosis with manual citations (RAG)",
            "diagnosis (RAG, v1)": "Detailed RAG diagnosis with thorough citations",
            "diagnosis (RAG, concise)": "Brief RAG diagnosis format",
            "diagnosis (RAG, safety_focused)": "Safety-first RAG diagnosis with manual citations",
            "repurposing (baseline)": "Identify repurposing scenarios (Dörnbach methodology)",
            "repurposing (baseline, detailed)": "Comprehensive scenarios with feasibility analysis",
            "repurposing (baseline, creative)": "Innovative and unconventional repurposing ideas",
            "repurposing (baseline, technical)": "Technical specifications and compatibility focus",
            "repurposing (baseline, concise)": "Quick list of practical repurposing scenarios",
            "repurposing (RAG)": "Repurposing scenarios from technical documentation",
            "repurposing (RAG, detailed)": "Comprehensive RAG scenarios with spec citations",
            "repurposing (RAG, creative)": "Creative RAG scenarios validated by docs",
            "repurposing (RAG, technical)": "Technical RAG scenarios with full specifications",
            "repurposing (RAG, concise)": "Quick RAG scenarios with citations",
            "repurposing_properties": "Extract technical properties for repurposing (RQ2)",
            "ml_recommendation (baseline)": "Recommend ML algorithms (Sonntag methodology)",
            "ml_recommendation (baseline, detailed)": "Comprehensive ML guidance with implementation details",
            "ml_recommendation (baseline, beginner_friendly)": "Simple explanations for non-experts",
            "ml_recommendation (baseline, technical)": "Rigorous analysis with mathematical formulations",
            "ml_recommendation (baseline, concise)": "Quick ML algorithm recommendations",
            "ml_recommendation (RAG)": "ML recommendations from knowledge base",
            "ml_recommendation (RAG, detailed)": "Comprehensive ML guidance from research papers",
            "ml_recommendation (RAG, beginner_friendly)": "Simple ML explanations from learning resources",
            "ml_recommendation (RAG, technical)": "Technical ML analysis from academic papers",
            "ml_recommendation (RAG, concise)": "Quick ML recommendations from knowledge base",
        }


# ================= Output Validators =================


class OutputValidator:
    """
    Validators for template outputs to ensure compliance with expected formats.

    Usage:
        validator = OutputValidator()
        is_valid, errors = validator.validate_ml_output(output_text, required_keys=['ML_problem_type'])
    """

    # Regex pattern for ML key:value format
    ML_KEY_VALUE_PATTERN = re.compile(r'^[A-Za-z0-9_]+:\s+.+$')

    @staticmethod
    def validate_ml_output(output: str, required_keys: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        r"""
        Validate ML recommendation output format.

        Rules:
        - Every non-empty line must match: ^[A-Za-z0-9_]+:\s+.+$
        - All required keys must be present

        Args:
            output: The LLM output to validate
            required_keys: Optional list of required key names (e.g., ['ML_problem_type', 'ML_algorithm_1'])

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        lines = output.strip().split('\n')
        found_keys = set()

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Check if line matches key:value format
            if not OutputValidator.ML_KEY_VALUE_PATTERN.match(line):
                errors.append(f"Line {i} does not match key:value format: '{line}'")
            else:
                # Extract key name
                key = line.split(':', 1)[0].strip()
                found_keys.add(key)

        # Check required keys
        if required_keys:
            missing_keys = set(required_keys) - found_keys
            if missing_keys:
                errors.append(f"Missing required keys: {sorted(missing_keys)}")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_repurposing_output(output: str, variant: str = "default") -> Tuple[bool, List[str]]:
        """
        Validate repurposing scenario output format.

        Rules:
        - Exactly 10 lines (non-empty)
        - Each line contains exactly one | (default) or two | (creative variant)
        - No numbering (lines should not start with digits followed by .)

        Args:
            output: The LLM output to validate
            variant: Template variant ("default", "creative", etc.)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        lines = [line.strip() for line in output.strip().split('\n') if line.strip()]

        # Check exactly 10 lines
        if len(lines) != 10:
            errors.append(f"Expected exactly 10 lines, got {len(lines)}")

        # Determine expected pipe count
        expected_pipes = 2 if variant.lower() == "creative" else 1

        for i, line in enumerate(lines, 1):
            # Check for numbering (e.g., "1. " or "1) ")
            if re.match(r'^\d+[\.\)]\s', line):
                errors.append(f"Line {i} contains numbering: '{line}'")

            # Check pipe count
            pipe_count = line.count('|')
            if pipe_count != expected_pipes:
                errors.append(
                    f"Line {i} contains {pipe_count} pipes, expected {expected_pipes}: '{line}'"
                )

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_repurposing_properties_output(output: str) -> Tuple[bool, List[str]]:
        """
        Validate repurposing properties output format.

        Rules:
        - One property per line
        - Each line should be a key:value pair (natural language format)
        - No strict format required (flexible for property descriptions)

        Args:
            output: The LLM output to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        lines = [line.strip() for line in output.strip().split('\n') if line.strip()]

        if len(lines) == 0:
            errors.append("Output is empty, expected at least one property")

        # Properties should generally have a colon to separate key from value
        for i, line in enumerate(lines, 1):
            if ':' not in line:
                # Warning only, not strict requirement
                errors.append(f"Line {i} may not be a valid property (no colon): '{line}'")

        return (len(errors) == 0, errors)


# ================= Custom Template Builder =================

class CustomPromptBuilder:
    """
    Builder for creating custom prompts that follow research methodology.

    Example:
        builder = CustomPromptBuilder()
        builder.set_background("You are an expert...")
        builder.set_problem("We need to...")
        builder.set_task("Please identify...")
        builder.set_format("Format: ...")
        template = builder.build()
    """

    def __init__(self):
        self.background = None
        self.problem = None
        self.task = None
        self.format = None
        self.context_template = None

    def set_background(self, text: str) -> 'CustomPromptBuilder':
        """Set background information section"""
        self.background = text
        return self

    def set_problem(self, text: str) -> 'CustomPromptBuilder':
        """Set problem description section"""
        self.problem = text
        return self

    def set_task(self, text: str) -> 'CustomPromptBuilder':
        """Set task description section"""
        self.task = text
        return self

    def set_format(self, text: str) -> 'CustomPromptBuilder':
        """Set format description section"""
        self.format = text
        return self

    def set_context_template(self, text: str) -> 'CustomPromptBuilder':
        """Set context template for RAG prompts"""
        self.context_template = text
        return self

    def build(self) -> PromptTemplate:
        """
        Build the prompt template.

        Returns:
            PromptTemplate instance

        Raises:
            ValueError: If required sections are missing
        """
        if not self.problem:
            raise ValueError("Problem description is required")
        if not self.task:
            raise ValueError("Task description is required")
        if not self.format:
            raise ValueError("Format description is required")

        return PromptTemplate(
            problem_description=self.problem,
            task_description=self.task,
            format_description=self.format,
            background_information=self.background,
            context_template=self.context_template
        )
