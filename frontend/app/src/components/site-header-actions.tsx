import DiscordSvg from '@/components/icons/discord.svg';
import GithubSvg from '@/components/icons/github.svg';
import TwitterXSvg from '@/components/icons/twitter-x.svg';
import { ThemeToggle } from '@/components/theme-toggle';
import { Button } from '@/components/ui/button';
import clsx from 'clsx';
import NextLink from 'next/link';

export type SiteSocialsType = {
  github?: string | null;
  twitter?: string | null;
  discord?: string | null;
};

export function SiteHeaderActions (props: {
  className?: string;
  social?: SiteSocialsType;
}) {
  const { className, social = {} } = props;
  return (
    <div className={clsx('h-header w-full gap-0.5 items-center', className)}>
      <ThemeToggle />
      {social?.github && (
        <NextLink href={social.github} target="_blank" className="ml-auto">
          <Button size="icon" variant="ghost" className="rounded-full">
            <GithubSvg />
          </Button>
        </NextLink>
      )}

      {social.twitter && (
        <NextLink href={social.twitter} target="_blank" className="">
          <Button size="icon" variant="ghost" className="rounded-full">
            <TwitterXSvg />
          </Button>
        </NextLink>
      )}

      {social.discord && (
        <NextLink href={social.discord} target="_blank" className="">
          <Button size="icon" variant="ghost" className="rounded-full">
            <DiscordSvg />
          </Button>
        </NextLink>
      )}
    </div>
  );
}
