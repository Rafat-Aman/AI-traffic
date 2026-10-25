#!/bin/bash
# ============================================================================
#  DUAL-REPO SYNC SCRIPT
#  Friend's repo (origin): https://github.com/isti155/AI-Traffic-Surveillance-System-for-Dhaka-City
#  Your repo    (myrepo):   https://github.com/Rafat-Aman/AI-traffic.git
# ============================================================================

set -e  # Exit on error (we selectively handle merge conflicts)

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header()  { echo -e "\n${CYAN}${BOLD}═══ $1 ═══${NC}\n"; }
print_success() { echo -e "${GREEN}✔ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error()   { echo -e "${RED}✖ $1${NC}"; }
print_info()    { echo -e "${CYAN}→ $1${NC}"; }

# ── Verify we're in a git repo ──────────────────────────────────────────────
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    print_error "Not inside a Git repository!"
    exit 1
fi

# ── Verify remotes exist ────────────────────────────────────────────────────
if ! git remote get-url origin &>/dev/null; then
    print_error "Remote 'origin' (friend's repo) not found!"
    exit 1
fi
if ! git remote get-url myrepo &>/dev/null; then
    print_error "Remote 'myrepo' (your repo) not found!"
    print_info "Run: git remote add myrepo https://github.com/Rafat-Aman/AI-traffic.git"
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)

# ── Menu ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║        DUAL-REPO SYNC TOOL                      ║"
echo "║  origin → isti155 (friend)                      ║"
echo "║  myrepo → Rafat-Aman (you)                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Current branch: ${BOLD}${CURRENT_BRANCH}${NC}"
echo ""
echo "  1) ${GREEN}Push${NC}       — Commit & push to BOTH repos"
echo "  2) ${CYAN}Fetch${NC}      — Fetch friend's latest & merge into your branch"
echo "  3) ${YELLOW}Sync All${NC}  — Fetch from friend + merge + push to both repos"
echo "  4) ${BOLD}Status${NC}     — Show status of both remotes"
echo "  5) Exit"
echo ""
read -p "  Choose [1-5]: " choice

case $choice in

# ═══════════════════════════════════════════════════════════════════════════
# 1) COMMIT & PUSH TO BOTH
# ═══════════════════════════════════════════════════════════════════════════
1)
    print_header "COMMIT & PUSH TO BOTH REPOS"

    # Show what's changed
    git status --short
    echo ""

    # Check if there are changes to commit
    if git diff --quiet && git diff --cached --quiet; then
        print_info "No local changes to commit."
        read -p "  Push existing commits anyway? [y/N]: " push_anyway
        if [[ "$push_anyway" != "y" && "$push_anyway" != "Y" ]]; then
            exit 0
        fi
    else
        # Stage and commit
        read -p "  Stage all changes? [Y/n]: " stage_all
        if [[ "$stage_all" != "n" && "$stage_all" != "N" ]]; then
            git add -A
            print_success "All changes staged."
        else
            print_info "Stage files manually, then re-run."
            exit 0
        fi

        read -p "  Commit message: " commit_msg
        if [[ -z "$commit_msg" ]]; then
            commit_msg="Update $(date '+%Y-%m-%d %H:%M')"
            print_warning "Using default message: $commit_msg"
        fi
        git commit -m "$commit_msg"
        print_success "Committed."
    fi

    # Push to friend's repo
    print_info "Pushing to origin (friend's repo)..."
    if git push origin "$CURRENT_BRANCH" 2>&1; then
        print_success "Pushed to origin/$CURRENT_BRANCH"
    else
        print_error "Failed to push to origin. You may need to pull first (use option 2 or 3)."
    fi

    # Push to your repo
    print_info "Pushing to myrepo (your repo)..."
    if git push myrepo "$CURRENT_BRANCH" 2>&1; then
        print_success "Pushed to myrepo/$CURRENT_BRANCH"
    else
        print_warning "Failed to push to myrepo. Trying with --force for first-time setup..."
        read -p "  Force push to your repo? [y/N]: " force_push
        if [[ "$force_push" == "y" || "$force_push" == "Y" ]]; then
            git push myrepo "$CURRENT_BRANCH" --force
            print_success "Force-pushed to myrepo/$CURRENT_BRANCH"
        fi
    fi

    echo ""
    print_success "Done! Both repos are up to date."
    ;;

# ═══════════════════════════════════════════════════════════════════════════
# 2) FETCH FROM FRIEND & MERGE
# ═══════════════════════════════════════════════════════════════════════════
2)
    print_header "FETCH FROM FRIEND'S REPO & MERGE"

    # Fetch latest from friend
    print_info "Fetching from origin (friend's repo)..."
    git fetch origin
    print_success "Fetched latest from origin."

    # Show available remote branches
    echo ""
    print_info "Remote branches on origin:"
    git branch -r --list 'origin/*'
    echo ""

    # Ask which branch to merge from
    read -p "  Merge from which branch? [Istiaque]: " merge_branch
    merge_branch=${merge_branch:-Istiaque}

    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_warning "You have uncommitted changes. Stashing them first..."
        git stash push -m "auto-stash before merge $(date '+%Y-%m-%d %H:%M')"
        STASHED=true
    else
        STASHED=false
    fi

    # Attempt merge
    print_info "Merging origin/$merge_branch into $CURRENT_BRANCH..."
    set +e  # Don't exit on merge conflict
    MERGE_OUTPUT=$(git merge "origin/$merge_branch" --no-edit 2>&1)
    MERGE_STATUS=$?
    set -e

    if [ $MERGE_STATUS -eq 0 ]; then
        print_success "Merge successful!"
        echo "$MERGE_OUTPUT"
    else
        echo "$MERGE_OUTPUT"
        echo ""
        print_error "MERGE CONFLICT DETECTED!"
        echo ""
        print_info "Conflicting files:"
        git diff --name-only --diff-filter=U
        echo ""
        echo -e "  ${BOLD}How to resolve:${NC}"
        echo "    1. Open the conflicting files and fix the conflicts"
        echo "       (look for <<<<<<< ======= >>>>>>> markers)"
        echo "    2. Stage resolved files:  git add <file>"
        echo "    3. Complete the merge:    git commit"
        echo ""
        echo "    OR abort the merge:       git merge --abort"
        echo ""

        read -p "  Open conflicting files in editor? [y/N]: " open_editor
        if [[ "$open_editor" == "y" || "$open_editor" == "Y" ]]; then
            CONFLICT_FILES=$(git diff --name-only --diff-filter=U)
            if command -v code &>/dev/null; then
                echo "$CONFLICT_FILES" | xargs code
            elif command -v nano &>/dev/null; then
                echo "$CONFLICT_FILES" | head -1 | xargs nano
            else
                echo "$CONFLICT_FILES" | head -1 | xargs vi
            fi
        fi

        read -p "  Have you resolved all conflicts? [y/N]: " resolved
        if [[ "$resolved" == "y" || "$resolved" == "Y" ]]; then
            git add -A
            git commit --no-edit
            print_success "Merge completed!"
        else
            print_warning "Merge is still in progress. Resolve conflicts and run: git commit"
        fi
    fi

    # Restore stashed changes
    if [ "$STASHED" = true ]; then
        print_info "Restoring your stashed changes..."
        set +e
        git stash pop
        POP_STATUS=$?
        set -e
        if [ $POP_STATUS -ne 0 ]; then
            print_warning "Stash pop had conflicts. Your stash is preserved. Run: git stash show"
        else
            print_success "Stashed changes restored."
        fi
    fi
    ;;

# ═══════════════════════════════════════════════════════════════════════════
# 3) FULL SYNC: FETCH + MERGE + PUSH TO BOTH
# ═══════════════════════════════════════════════════════════════════════════
3)
    print_header "FULL SYNC — FETCH, MERGE & PUSH"

    # Step 1: Stash if dirty
    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_warning "Stashing uncommitted changes..."
        git stash push -m "auto-stash before sync $(date '+%Y-%m-%d %H:%M')"
        STASHED=true
    else
        STASHED=false
    fi

    # Step 2: Fetch from friend
    print_info "Step 1/4: Fetching from origin (friend)..."
    git fetch origin
    print_success "Fetched."

    # Step 3: Merge friend's default branch
    read -p "  Merge from which branch? [Istiaque]: " merge_branch
    merge_branch=${merge_branch:-Istiaque}

    print_info "Step 2/4: Merging origin/$merge_branch into $CURRENT_BRANCH..."
    set +e
    MERGE_OUTPUT=$(git merge "origin/$merge_branch" --no-edit 2>&1)
    MERGE_STATUS=$?
    set -e

    if [ $MERGE_STATUS -ne 0 ]; then
        echo "$MERGE_OUTPUT"
        print_error "MERGE CONFLICT! Resolve conflicts before continuing."
        echo ""
        print_info "Conflicting files:"
        git diff --name-only --diff-filter=U
        echo ""
        echo "  Resolve conflicts, then run:"
        echo "    git add -A && git commit"
        echo "    ./sync.sh   (choose option 1 to push)"
        echo ""

        # Restore stash before exiting
        if [ "$STASHED" = true ]; then
            print_info "Your uncommitted changes are stashed. Run 'git stash pop' after resolving."
        fi
        exit 1
    fi
    print_success "Merged successfully."

    # Step 4: Push to friend's repo
    print_info "Step 3/4: Pushing to origin (friend)..."
    if git push origin "$CURRENT_BRANCH" 2>&1; then
        print_success "Pushed to origin/$CURRENT_BRANCH"
    else
        print_warning "Could not push to origin (you may not have write access to all branches)."
    fi

    # Step 5: Push to your repo
    print_info "Step 4/4: Pushing to myrepo (your repo)..."
    if git push myrepo "$CURRENT_BRANCH" 2>&1; then
        print_success "Pushed to myrepo/$CURRENT_BRANCH"
    else
        print_warning "First time pushing to myrepo? Trying force push..."
        read -p "  Force push to your repo? [y/N]: " force_push
        if [[ "$force_push" == "y" || "$force_push" == "Y" ]]; then
            git push myrepo "$CURRENT_BRANCH" --force
            print_success "Force-pushed to myrepo/$CURRENT_BRANCH"
        fi
    fi

    # Restore stash
    if [ "$STASHED" = true ]; then
        print_info "Restoring stashed changes..."
        set +e
        git stash pop
        set -e
    fi

    echo ""
    print_success "Full sync complete! Both repos are up to date."
    ;;

# ═══════════════════════════════════════════════════════════════════════════
# 4) STATUS
# ═══════════════════════════════════════════════════════════════════════════
4)
    print_header "REMOTE STATUS"

    echo -e "${BOLD}Remotes:${NC}"
    git remote -v
    echo ""

    echo -e "${BOLD}Current branch:${NC} $CURRENT_BRANCH"
    echo ""

    echo -e "${BOLD}Local status:${NC}"
    git status --short
    echo ""

    # Fetch to compare
    print_info "Fetching latest info..."
    git fetch --all --quiet 2>/dev/null

    echo -e "${BOLD}Commits ahead/behind origin/$CURRENT_BRANCH:${NC}"
    if git rev-parse "origin/$CURRENT_BRANCH" &>/dev/null; then
        AHEAD=$(git rev-list --count "origin/$CURRENT_BRANCH..HEAD" 2>/dev/null || echo "?")
        BEHIND=$(git rev-list --count "HEAD..origin/$CURRENT_BRANCH" 2>/dev/null || echo "?")
        echo -e "  ↑ Ahead:  ${GREEN}$AHEAD${NC} commits"
        echo -e "  ↓ Behind: ${YELLOW}$BEHIND${NC} commits"
    else
        print_warning "Branch '$CURRENT_BRANCH' not found on origin."
    fi

    echo ""
    echo -e "${BOLD}Commits ahead/behind myrepo/$CURRENT_BRANCH:${NC}"
    if git rev-parse "myrepo/$CURRENT_BRANCH" &>/dev/null; then
        AHEAD=$(git rev-list --count "myrepo/$CURRENT_BRANCH..HEAD" 2>/dev/null || echo "?")
        BEHIND=$(git rev-list --count "HEAD..myrepo/$CURRENT_BRANCH" 2>/dev/null || echo "?")
        echo -e "  ↑ Ahead:  ${GREEN}$AHEAD${NC} commits"
        echo -e "  ↓ Behind: ${YELLOW}$BEHIND${NC} commits"
    else
        print_warning "Branch '$CURRENT_BRANCH' not found on myrepo (push first with option 1)."
    fi

    echo ""
    echo -e "${BOLD}Recent commits:${NC}"
    git log --oneline -5
    ;;

# ═══════════════════════════════════════════════════════════════════════════
# 5) EXIT
# ═══════════════════════════════════════════════════════════════════════════
5)
    echo "Bye!"
    exit 0
    ;;

*)
    print_error "Invalid option. Choose 1-5."
    exit 1
    ;;
esac
