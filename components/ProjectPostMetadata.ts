export interface ProjectPostMetadata {
    title: string;
    date: string;
    subtitle: string;
    slug: string;
    // images: string[]; // Array of image paths
    producturl: string;
    media: {
        type: 'image' | 'video';
        url: string;
        thumbnail?: string; // Optional thumbnail for videos  
        link?: string;
    }[];
}