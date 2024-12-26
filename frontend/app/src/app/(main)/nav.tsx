'use client';

import { logout } from '@/api/auth';
import type { PublicWebsiteSettings } from '@/api/site-settings';
import { useAuth } from '@/components/auth/AuthProvider';
import { Branding } from '@/components/branding';
import { useAllChatEngines } from '@/components/chat-engine/hooks';
import { ChatNewDialog } from '@/components/chat/chat-new-dialog';
import { ChatsHistory } from '@/components/chat/chats-history';
import { useAllKnowledgeBases } from '@/components/knowledge-base/hooks';
import { type NavGroup, SiteNav } from '@/components/site-nav';
import { useBootstrapStatus } from '@/components/system/BootstrapStatusProvider';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader } from '@/components/ui/sidebar';
import { Skeleton } from '@/components/ui/skeleton';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useHref } from '@/components/use-href';
import { ActivitySquareIcon, AlertTriangleIcon, BinaryIcon, BotMessageSquareIcon, BrainCircuitIcon, CircleDotIcon, CogIcon, ComponentIcon, FileLineChart, HomeIcon, KeyRoundIcon, LibraryBigIcon, LibraryIcon, LogInIcon, MessageCircleQuestionIcon, MessagesSquareIcon, ShuffleIcon } from 'lucide-react';
import NextLink from 'next/link';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import type { ReactNode } from 'react';

export function SiteSidebar ({ setting }: { setting: PublicWebsiteSettings }) {
  return (
    <Sidebar>
      <SidebarHeader>
        <Branding setting={setting} />
      </SidebarHeader>
      <SidebarContent>
        <NavContent />
      </SidebarContent>
      <SidebarFooter>
        <NavFooter />
      </SidebarFooter>
    </Sidebar>
  );
}

function NavContent () {
  const { required, need_migration } = useBootstrapStatus();
  const href = useHref();
  const auth = useAuth();
  const user = auth.me;
  const isLoggedIn = !!user; // TODO: wait for server

  const disableIfNotAuthenticated = !isLoggedIn ? <><Link className="font-semibold underline" href={`/auth/login?callbackUrl=${encodeURIComponent(href)}`}>Login</Link> to continue</> : false;

  const groups: NavGroup[] = [
    {
      items: [
        { custom: true, key: 'new-chat', children: <ChatNewDialog /> },
        { href: '/', title: 'Home', icon: HomeIcon, exact: true },
        { href: '/c', title: 'Conversations', exact: true, icon: MessagesSquareIcon, disabled: disableIfNotAuthenticated },
        { custom: true, key: 'history', children: <ChatsHistory /> },
      ],
    },
  ];

  if (user?.is_superuser) {
    groups.push({
      title: 'Admin',
      items: [
        { href: '/stats/trending', title: 'Dashboard', icon: ActivitySquareIcon },
        {
          href: '/knowledge-bases',
          title: 'Knowledge Bases',
          icon: LibraryBigIcon,
          details: !required.knowledge_base
            ? <NavWarningDetails>You need to configure at least one Knowledge Base.</NavWarningDetails>
            : <KnowledgeBaseNavDetails />,
        },
        {
          href: '/chat-engines',
          title: 'Chat Engines',
          icon: BotMessageSquareIcon,
          details: !!need_migration.chat_engines_without_kb_configured?.length
            ? <NavWarningDetails>
              Some ChatEngine need to <a href="/releases/0.3.0#manual-migration" className="underline">configure KnowledgeBase</a>.
            </NavWarningDetails>
            : !required.default_chat_engine
              ? <NavWarningDetails>You need to configure default Chat Engine.</NavWarningDetails>
              : <ChatEnginesNavDetails />,
        },
        {
          parent: true,
          key: 'models',
          title: 'Models',
          icon: ComponentIcon,
          details: (!required.default_llm || !required.default_embedding_model) && <NavWarningDetails />,
          children: [
            { href: '/llms', title: 'LLMs', icon: BrainCircuitIcon, details: !required.default_llm ? <NavWarningDetails>You need to configure at least one Default LLM.</NavWarningDetails> : undefined },
            { href: '/embedding-models', title: 'Embedding Models', icon: BinaryIcon, details: !required.default_embedding_model && <NavWarningDetails>You need to configure at least one Default Embedding Model.</NavWarningDetails> },
            { href: '/reranker-models', title: 'Reranker Models', icon: ShuffleIcon },
          ],
        },
        { href: '/feedbacks', title: 'Feedbacks', icon: MessageCircleQuestionIcon },
        {
          parent: true,
          key: 'evaluation',
          title: 'Evaluation',
          icon: FileLineChart,
          children: [
            { href: '/evaluation/tasks', title: 'Tasks', icon: CircleDotIcon },
            { href: '/evaluation/datasets', title: 'Datasets', icon: LibraryIcon },
          ],
        },
        { href: '/site-settings', title: 'Settings', icon: CogIcon },
      ],
      sectionProps: { className: 'mt-auto mb-0' },
    });
  }

  if (user?.is_superuser) {
    groups.push({
      title: 'Account',
      items: [
        { href: '/api-keys', title: 'API Keys', icon: KeyRoundIcon },
      ],
    });
  }

  return (
    <SiteNav groups={groups} />
  );
}

function NavFooter () {
  const href = useHref();
  const user = useAuth().me;
  const router = useRouter();

  if (!user) {
    return (
      <Button variant="ghost" asChild>
        <NextLink href={`/auth/login?callbackUrl=${encodeURIComponent(href)}`} prefetch={false} className="items-center w-full gap-2">
          <LogInIcon size="1em" />
          Login
        </NextLink>
      </Button>
    );
  }
  return (
    <div className="flex items-center gap-2">
      <DropdownMenu>
        <DropdownMenuTrigger>
          <Avatar className="border dark:bg-primary bg-primary-foreground p-0.5 w-8 h-8">
            {/*{user.image && <AvatarImage src={user.image} />}*/}
            <AvatarFallback className="text-xs">
              {/*{user.image ? <Skeleton className="w-full h-full" /> : user.name}*/}
              {user.email.slice(0, 2)}
            </AvatarFallback>
          </Avatar>
        </DropdownMenuTrigger>
        <DropdownMenuContent collisionPadding={8} side="top">
          <DropdownMenuItem onClick={() => {
            logout().finally(() => {
              router.refresh();
            });
          }}>
            Sign out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <span className="text-sm font-semibold">
        {user.email}
      </span>
    </div>
  );
}

function NavWarningDetails ({ children }: { children?: ReactNode }) {
  if (!children) {
    return <AlertTriangleIcon className="text-warning size-4" />;
  }
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger>
          <AlertTriangleIcon className="text-warning size-4" />
        </TooltipTrigger>
        <TooltipContent>
          {children}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function CountSpan ({ children }: { children?: ReactNode }) {
  return <span className="text-xs opacity-50 font-normal inline-block mr-1">{children}</span>;
}

function KnowledgeBaseNavDetails () {
  const { data: knowledgeBases, isLoading } = useAllKnowledgeBases();

  if (isLoading) {
    return <Skeleton className="flex-shrink-0 w-6 h-4" />;
  }

  return <CountSpan>{knowledgeBases?.length}</CountSpan>;
}

function ChatEnginesNavDetails () {
  const { data, isLoading } = useAllChatEngines();

  if (isLoading) {
    return <Skeleton className="flex-shrink-0 w-6 h-4" />;
  }

  return <CountSpan>{data?.length}</CountSpan>;
}
