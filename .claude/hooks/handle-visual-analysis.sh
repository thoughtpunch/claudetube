#!/bin/bash
# Claude Code hook: Post-processing reminder for visual analysis
#
# This hook fires after process_video_tool completes and checks if
# visual analysis is strongly recommended. If so, it injects a
# system reminder to encourage Claude to extract frames.

# Read the hook input from stdin
INPUT=$(cat)

# The PostToolUse event provides tool_response as the actual output
# Parse the tool response - it's the JSON result from process_video_tool
RESPONSE="$INPUT"

# Check if visual_analysis.strongly_recommended is true
# Use -e flag to exit with non-zero if condition is false
if echo "$RESPONSE" | jq -e '.visual_analysis.strongly_recommended == true' > /dev/null 2>&1; then
    # Extract details for the reminder
    SCORE=$(echo "$RESPONSE" | jq -r '.visual_analysis.score // "unknown"')
    REASONING=$(echo "$RESPONSE" | jq -r '.visual_analysis.reasoning // "Visual content detected"')
    ELEMENTS=$(echo "$RESPONSE" | jq -r '.visual_analysis.likely_elements | join(", ") // ""')

    # Build the context message
    CONTEXT="## VISUAL ANALYSIS STRONGLY RECOMMENDED

**Score:** ${SCORE}/10
**Reasoning:** ${REASONING}"

    if [ -n "$ELEMENTS" ] && [ "$ELEMENTS" != "null" ]; then
        CONTEXT="${CONTEXT}
**Expected visual elements:** ${ELEMENTS}"
    fi

    CONTEXT="${CONTEXT}

This video has HIGH VISUAL INFORMATION DENSITY. The visuals ARE the explanation.
**Before synthesizing content**, use \`get_frames\` or \`get_hq_frames\` to extract key frames.

Consider extracting frames at:
- The beginning (overview/intro visuals)
- Key concept explanations (diagrams, animations)
- Code demonstrations (syntax, output)
- Summary sections"

    # Output JSON with additionalContext
    # Using jq to properly escape the context string
    echo "$CONTEXT" | jq -R -s '{
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": .
        }
    }'
    exit 0
elif echo "$RESPONSE" | jq -e '.visual_analysis.recommended == true' > /dev/null 2>&1; then
    # Visual analysis is recommended (but not required)
    SCORE=$(echo "$RESPONSE" | jq -r '.visual_analysis.score // "unknown"')

    CONTEXT="Visual analysis recommended (score: ${SCORE}/10). Consider using \`get_frames\` if visual context would improve understanding."

    echo "$CONTEXT" | jq -R -s '{
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": .
        }
    }'
    exit 0
else
    # Visual analysis not needed or error parsing - exit silently
    exit 0
fi
